import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict
from typing import Final
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import attr
from loguru import logger
from utils.events import get_expiration_event
from utils.run_command import CommandRunner
from utils.run_command import FullConnection

EXPIRATION_SECONDS: Final[float] = 100.0


BW_ERROR_VALUE: Final[float] = 0.0
LAT_ERROR_VALUE: Final[float] = 10000000

USE_GDR: Final[bool] = True
BW_LOWER_LIMIT_GDR: Final[float] = 720
BW_LOWER_LIMIT_NO_GDR: Final[float] = 300

LAT_UPPER_LIMIT: Final[float] = 4.2

BW_TEST_OUTPUT_KEY = "BWaverage[Gb/sec]"
LAT_TEST_OUTPUT_KEY = "99%percentile[usec]"

IP = str


@attr.s(auto_attribs=True, frozen=True)
class HcaDescription:
    pcie_device_description: str

    @property
    def pcie_slot_index(self) -> int:
        return int(self.pcie_device_description.split("_")[1])

    def get_gpu_index(self) -> int:
        return MLX_CARDS.index(self.pcie_device_description)

    def __str__(self) -> str:
        return self.pcie_device_description


EXPECTED_VERSION_FILE = Path(os.path.realpath(__file__)).parent.parent / "health_checks" / "config.json"
with open(EXPECTED_VERSION_FILE, 'r') as f:
    EXPECTED_VERSIONS = json.load(f)

MLX_CARDS: Final[Tuple[HcaDescription, ...]] = tuple(HcaDescription(device_name) for device_name in EXPECTED_VERSIONS["infiniband_status"]["device_names"])


def is_passing_host(card_to_result: Dict[str, Tuple[float, float]], gdr_enabled: bool = USE_GDR) -> bool:
    if gdr_enabled:
        bw_lower_limit = BW_LOWER_LIMIT_GDR
    else:
        bw_lower_limit = BW_LOWER_LIMIT_NO_GDR
    for card, (bw, lat) in card_to_result.items():
        if bw < bw_lower_limit or lat > LAT_UPPER_LIMIT:
            return False
    return True


def find_good_hosts(
    connections_to_result: Dict[str, Dict[str, Tuple[float, float]]], gdr_enabled: bool = USE_GDR
) -> List[str]:
    good_hosts = []
    for connection, result in connections_to_result.items():
        if is_passing_host(result, gdr_enabled):
            good_hosts.append(connection)
    return good_hosts


def parse_p2p_output(uncleaned_output: str, key: str) -> Optional[float]:
    """
    The p2p output is terrible:
    - The headers are not separated by tabs, but by variable numbers of spaces.
    - The header values may themselves contain spaces.
    - The --output=json option produces invalid JSON.

    As a result, we have some nasty parsing logic here; see the unit tests for illustrative
    examples.

    If/when there is a better way to extract the desired information, we should use it.
    """
    split_text = re.split("-+", uncleaned_output)
    data_values = split_text[-2].strip()
    # Change all the headers to not have spaces within them
    data_values = data_values.replace("% percentile", "%percentile")
    data_values = data_values.replace("BW ", "BW")
    data_values = re.sub("Conflicting CPU frequency.*", "", data_values)
    lines = [l for l in data_values.splitlines() if len(l.strip()) > 0]
    headers = [x.strip() for x in re.split(r"\s+", lines[0]) if len(x.strip()) > 0]
    values = [x.strip() for x in re.split(r"\s+", lines[1]) if len(x.strip()) > 0]
    for header, val in zip(headers, values):
        if header == key:
            return float(val)
    raise ValueError(f"Could not find key {key} in output {uncleaned_output}, output format may have changed")


def _build_ib_write_bw_command(
    card: HcaDescription,
    iters: int,
    port: int,
    use_gdr: bool,
    other_ip: Optional[str] = None,
) -> str:
    return " ".join(
        (
            "ib_write_bw",
            "-b",
            f"-d {card}",
            *((f"--use_cuda", str(card.get_gpu_idx())) if use_gdr else ()),
            *((other_ip,) if other_ip is not None else ()),
            f"--iters {iters}",
            f"-p {port}",
            "--report_gbits",
        )
    )


def shutdown_test(connection: CommandRunner, command: str) -> None:
    if "[" not in command:
        # This is to escape the command, so we don't end up killing the pkill before it kills the process we care about
        command = "[" + command[0] + "]" + command[1:]
    tries = 0
    max_retries = 10
    while True:
        running_commands_count = int(connection.run_command(f"ps aux | grep {command} | wc -l ").output.strip())
        if running_commands_count == 0:
            break
        try:
            connection.run_command(f"pkill -f {command}")
            logger.info(f"killed {command} on {connection} on try {tries}")
        except:
            pass
        tries += 1
        if tries >= max_retries:
            break
    logger.info(f"failed to kill {command} on {connection} after {max_retries} tries")


def run_single_rail_test(
    connection: CommandRunner,
    other_ip: str,
    is_head: bool,
    gpu_idx_and_card: Tuple[int, HcaDescription],
    same_host: bool = False,
    iters: int = 5_000,
) -> Tuple[Union[CommandRunner, str], str, Tuple[float, float]]:
    gpu_idx, card = gpu_idx_and_card
    bw_output, lat_output = BW_ERROR_VALUE, LAT_ERROR_VALUE
    try:
        if is_head:
            # Ensure the other card acting as a server has time to spin up
            time.sleep(5)
        with get_expiration_event(EXPIRATION_SECONDS) as event:
            other_ip = other_ip if is_head else ""
            if same_host:
                port = 18515 + int(card.pcie_slot_index) % 6
            else:
                port = 18515 + int(card.pcie_slot_index)
            command = _build_ib_write_bw_command(
                card=card,
                other_ip=other_ip,
                iters=iters,
                port=port,
                use_gdr=USE_GDR,
            )
            bw_result = connection.run_command(command, shutdown_event=event)
            if bw_result.returncode == 0:
                bw_output = parse_p2p_output(bw_result.output, key=BW_TEST_OUTPUT_KEY)
            else:
                logger.info(
                    f"Trying to kill ib_write_bw on {connection.ip}:{card} with {bw_result.returncode} {bw_result.output}"
                )
                shutdown_test(connection, f"'ib_write_b[w] -d {card}'")
        if is_head:
            # Ensure the other card acting as a server has time to spin up
            time.sleep(5)
        with get_expiration_event(EXPIRATION_SECONDS) as event:
            other_ip = other_ip if is_head else ""
            if same_host:
                port = 18514 - int(card.split("_")[1]) % 6
            else:
                port = 18514 - int(card.split("_")[1])
            # Perftest supports CUDA latency tests with read/send verbs only
            command = f"ib_write_lat -d {card} {other_ip} --iters {iters} -p {port}"
            lat_result = connection.run_command(command, shutdown_event=event)
            if lat_result.returncode == 0:
                lat_output = parse_p2p_output(lat_result.output, key=LAT_TEST_OUTPUT_KEY)
            else:
                logger.info(
                    f"Trying to kill ib_write_lat on {connection.ip}:{card} with {lat_result.returncode} {lat_result.output}"
                )
                shutdown_test(connection, f"'ib_write_[l]at -d {card}'")
        logger.info(f"Results for {connection.ip}:{card} bw: {bw_output} lat: {lat_output}")
        return connection, card, (bw_output, lat_output)
    except Exception as e:
        # We add square brackets around the w such that we avoid killing the `pkill` command itself?
        shutdown_test(connection, f"'ib_write_[l]at -d {card}'")
        shutdown_test(connection, f"'ib_write_b[w] -d {card}'")
        logger.info(f"caught exception on {connection}:\n{e}")
        return connection, card, (bw_output, lat_output)


def run_p2p_ib_test(
    connection: Union[CommandRunner, str], other_ip: str, is_head: True
) -> Tuple[str, Dict[str, Tuple[float, float]]]:
    card_to_result = {}

    for gpu_idx, card in enumerate(MLX_CARDS):
        _, _, card_result = run_single_rail_test(connection, other_ip, is_head, (gpu_idx, card), same_host=False)
        card_to_result[card] = card_result
    return connection.ip, card_to_result


def run_single_p2p(
    run: int,
    connections: Sequence[FullConnection],
    output_file: Optional[Path] = None,
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    connection_pairs = list(zip(connections[: len(connections) // 2], connections[len(connections) // 2 :]))
    servers = [(pair[0].ssh_connection, pair[1].internal_ip, False) for pair in connection_pairs]
    clients = [(pair[1].ssh_connection, pair[0].internal_ip, True) for pair in connection_pairs]
    alternating_server_client = [item for pair in zip(servers, clients) for item in pair]
    connection_to_result = {}
    with ThreadPoolExecutor(max_workers=len(alternating_server_client)) as executor:
        results = executor.map(
            lambda group_and_card: run_p2p_ib_test(
                connection=group_and_card[0],
                other_ip=group_and_card[1],
                is_head=group_and_card[2],
            ),
            alternating_server_client,
        )
        for connection, result in results:
            if output_file:
                with output_file.open("a+") as f:
                    f.write(f"Results for {connection} in run {run}: {result}\n")
            connection_to_result[connection] = result

    return connection_to_result


def run_single_host_p2p(
    run: int,
    full_connections: Sequence[FullConnection],
    output_file: Optional[Path] = None,
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    first_half_cards = MLX_CARDS[: len(MLX_CARDS) // 2]
    second_half_cards = MLX_CARDS[len(MLX_CARDS) // 2 :]
    servers = [
        (connection.ssh_connection, connection.internal_ip, False, (gpu_idx, driver))
        for gpu_idx, driver in enumerate(first_half_cards)
        for connection in full_connections
    ]
    clients = [
        (connection.ssh_connection, connection.internal_ip, True, (gpu_idx + 4, driver))
        for gpu_idx, driver in enumerate(second_half_cards)
        for connection in full_connections
    ]
    alternating_server_client = [item for pair in zip(servers, clients) for item in pair]
    max_workers = len(alternating_server_client)

    connection_to_result = {connection.ssh_connection.ip: dict() for connection in full_connections}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        run_results = executor.map(
            lambda group_and_card: run_single_rail_test(
                connection=group_and_card[0],
                other_ip=group_and_card[1],
                is_head=group_and_card[2],
                gpu_idx_and_card=group_and_card[3],
                same_host=True,
            ),
            alternating_server_client,
        )
        for connection, card, result in run_results:
            if output_file:
                with output_file.open("a+") as f:
                    f.write(f"Results for {connection} {card} in run {run}: {result}\n")
            connection_to_result[connection.ip][card] = result

    logger.info(f"Finished running run {run} with {connection_to_result}")

    return connection_to_result


def run_p2p_ib_tests(
    connections: Sequence[CommandRunner],
    output_file: Optional[Path] = None,
    single_host: bool = False,
    num_iterations: int = 15,
) -> Dict[str, Dict[str, int]]:
    test = "p2p_ib" if not single_host else "host_p2p_ib"
    ip_to_runs_passed: Dict[str, int] = {connection.ip: 0 for connection in connections}
    if output_file:
        with output_file.open("a+") as f:
            f.write(f"Starting {test} test with connections {connections}\n")

    full_connections = [FullConnection(ssh_connection=c, internal_ip="127.0.0.1") for c in connections]

    last_results = None
    for run_count in range(num_iterations):
        try:
            local_rng = random.Random()
            local_rng.seed(run_count)
            run_count += 1
            if not single_host:
                mixed_nodes = local_rng.sample(full_connections, len(full_connections))
                connection_to_result = run_single_p2p(run_count, mixed_nodes, output_file)
            else:
                connection_to_result = run_single_host_p2p(run_count, full_connections, output_file)
            good_hosts = find_good_hosts(connection_to_result)
            for host in good_hosts:
                if host not in ip_to_runs_passed:
                    raise ValueError(f"Host {host} not in ip_to_runs_passed")
                ip_to_runs_passed[host] += 1
            last_results = connection_to_result
            bad_hosts = [connection.ip for connection in connections if connection.ip not in good_hosts]
            bad_hosts_results = {ip: last_results.get(ip, (BW_ERROR_VALUE, LAT_ERROR_VALUE)) for ip in bad_hosts}
            logger.info(f"Bad p2p_hosts {bad_hosts} with results: {bad_hosts_results}")
            logger.info(
                f"{test} after {run_count} iterations results: {sorted(ip_to_runs_passed.items(), key = lambda item: item[1])}"
            )
            if output_file:
                with output_file.open("a+") as f:
                    f.write(f"All results for run {run_count}: {connection_to_result}\n")
                    f.write(f"Bad p2p_hosts {bad_hosts} with results: {bad_hosts_results}\n")
                    f.write(
                        f"{test} after {run_count} iterations results: {sorted(ip_to_runs_passed.items(), key=lambda item: item[1])}\n"
                    )
        finally:
            for connection in connections:
                shutdown_test(connection, "ib_writ[e]_")
        # Wait a little after the tests to ensure everything can be cleaned up correctly
        time.sleep(5)

    logger.info(f"From last run all p2p_hosts results: {last_results}\n")
    logger.info(f"Final p2p_host results: {sorted(ip_to_runs_passed.items(), key = lambda item: item[1])}")

    if output_file:
        with output_file.open("a+") as f:
            f.write(f"From last run all p2p_hosts results: {last_results}")
            f.write(f"Final p2p_host results: {sorted(ip_to_runs_passed.items(), key=lambda item: item[1])}")

    ip_to_metrics = {
        ip: {"passes": passes, "count": num_iterations, "ratio": passes / num_iterations}
        for ip, passes in ip_to_runs_passed.items()
    }
    return ip_to_metrics
