"""
Basic procedure for using this script:

When you want to identify ports that have been misbehaving in the UFM event logs, run the subcommand `process-logs`:
```
$ python ufm_events/find_problematic_events.py process-logs ~/actions.jsonl
```

This will produce a file called `actions.jsonl` that contains a list of actions to take to disable the misbehaving ports.

In addition to the UFM event logs, we have found several other sources of information to be helpful:

You can run
```
sudo iblinkinfo | tee IBLINK_OUTPUT_PATH
python ufm_events/find_problematic_events.py get-bad-from-iblinkinfo IBLINK_OUTPUT_PATH OUTPUT_PATH
```

to get bad ports based on the results of the iblinkinfo command.

Similarly, if you have previously performed an IB burn, you can run
```
python ufm_events/find_problematic_events.py get-bad-from-counters
```

on the results of the burn.
"""

from __future__ import annotations

import argparse
import datetime
import functools
import os
import re
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Dict
from typing import Final
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import Self
from typing import Set
from typing import TextIO
from typing import Tuple
from typing import TypeAlias
from typing import TypeVar
from typing import Union
from zoneinfo import ZoneInfo

import attr
import requests
import urllib3

ERROR_FILE: TextIO = sys.stderr


urllib3.disable_warnings()
UFM_LOCAL_IP: Final[str] = "UFM_LOCAL_IP"
UFM_AUTH: Final[Tuple[str, str]] = ("USERNAME", "PASSWORD")


@functools.cache
def get_ports_by_port_id() -> Dict[PortId, Dict]:
    result = {}
    port_list = requests.get(f"https://{UFM_LOCAL_IP}/ufmRest/resources/ports", auth=UFM_AUTH, verify=False).json()
    for port in port_list:
        node_description = port["node_description"]
        try:
            switch, port_str = node_description.split(":")
        except ValueError:
            # Some node descriptions that do not describe switches do not contain colons.
            # For example: "MT1111 ConnectX7   Mellanox Technologies"
            continue
        port_id: PortId
        if port_str.count("/") == 1:
            port_id = PortId(switch=switch, port=port_str)
        elif port_str.count("/") == 2:
            port_str = port_str.removeprefix("1/")
            port_id = PortId(switch=switch, port=port_str)
        elif port_str.count("/") == 0:
            # Some ports now have the number format of ports
            port_id = get_portid(switch, int(port_str))
        else:
            raise ValueError(f"Unexpected port description: {port_str}")
        result[port_id] = port
    return result

Action: TypeAlias = Union["DisablePortAction", "ReenablePortAction"]

ActionT = TypeVar("ActionT", bound=Action)

R = TypeVar("R")


@attr.s(auto_attribs=True, frozen=True)
class DisablePortAction:
    """An action to disable a port."""

    port: PortId
    cause: PortRelatedEvent

    def __str__(self) -> str:
        return f"Disabling {self.port} due to {self.cause}"


@attr.s(auto_attribs=True, frozen=True)
class ReenablePortAction:
    """An action to re-enable a port."""

    port: PortId

    def __str__(self) -> str:
        return f"Re-enabling {self.port}"


ENTRY_REGEX: Final[re.Pattern] = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+) \[(?P<entry_id>\d+)\] \[(?P<message_code>\d+)\] (?P<level>\w+) \[(?P<topic>\w+)\] (?P<device_type>\w+) \[(?P<device_description>[^\]]+)\]( \[dev_id: (?P<device_id>\w+)\])?: (?P<message>.*)$"
)


def parse_pacific(s: str) -> datetime.datetime:
    """Parse a timestamp and set the timezone to Pacific time.

    At the time of writing, the timestamps in the event log are in Pacific time and look like this:
    2023-12-20 07:46:24.442
    """
    return datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=ZoneInfo("America/Los_Angeles"))


@attr.s(auto_attribs=True, frozen=True)
class Entry:
    timestamp: datetime.datetime
    message_code: int
    level: str
    device_type: str
    device_description: str
    device_id: Optional[str]
    message: str
    original_line: str = attr.field(repr=False)

    @classmethod
    def build_from_line(cls, line: str) -> Self:
        maybe_match = ENTRY_REGEX.match(line)
        if maybe_match is None:
            raise ValueError(f"Line does not match regex: {line}")

        return cls(
            timestamp=parse_pacific(maybe_match.group("timestamp")),
            message_code=int(maybe_match.group("message_code")),
            level=maybe_match.group("level"),
            device_type=maybe_match.group("device_type"),
            device_description=maybe_match.group("device_description"),
            device_id=maybe_match.group("device_id"),
            message=maybe_match.group("message"),
            original_line=line,
        )


THRESHOLD_EXCEEDED_CODES: Final[Tuple[int, ...]] = (
    110,  # Symbol error counter rate
    112,  # Link downed
    113,  # Port receive errors
    115,  # Receive switch relay errors
    116,  # Transmit discards
)

# Example message: Link went down: (Switch:T3-E21-N-U47:1/17/1)9c0591030090e000:33 - (Switch:T4-E45-L-U51:1/4/2)fc6a1c0300244e00:8, cable S/N: MT2330FT09118
LINK_WENT_DOWN_MESSAGE_PATTERN: Final[re.Pattern] = re.compile(
    r"^Link went down: \(Switch:(?P<switch>[^:]+):(1/)?(?P<port>[^)]+)\)[^(]+\(Switch:(?P<peer_switch>[^:]+):(1/)?(?P<peer_port>[^)]+)\)"
)


# Example message: Peer Port T2-E63-L-U37:1/19/2 is considered by SM as unhealthy due to FLAPPING.
PORT_IS_CONSIDERED_UNHEALTHY_MESSAGE_PATTERN: Final[re.Pattern] = re.compile(
    r"^Peer Port (?P<switch>[^:]+):(1/)?(?P<port>[^ ]+) is considered by SM as unhealthy due to \w+\."
)


def standardize_ports(ports: List[PortId]) -> List[PortId]:
    for i, port in enumerate(ports):
        if "/" not in port.port:
            ports[i] = get_portid(port.switch, int(port.port))
    return ports


@attr.s(auto_attribs=True, frozen=True)
class PortRelatedEvent:
    timestamp: datetime.datetime
    ports: Tuple[PortId, ...]
    original_line: str

    def __str__(self) -> str:
        return f"{self.timestamp}: {self.original_line} affecting {self.ports}"

    @classmethod
    def build_from_threshold_exceeded_entry(cls, entry: Entry) -> Self:
        assert (
            entry.message_code in THRESHOLD_EXCEEDED_CODES
        ), "Entry must be a threshold exceeded event: {THRESHOLD_EXCEEDED_CODES}"
        ports = [PortId.build_from_device_description(entry.device_description)]
        try:
            ports.append(PortId.build_from_device_description(entry.message.rstrip(".")))
        except ValueError:
            pass
        ports = standardize_ports(ports)
        return cls(timestamp=entry.timestamp, ports=tuple(sorted(ports, key=str)), original_line=entry.original_line)

    @classmethod
    def build_from_link_went_down_entry(cls, entry: Entry) -> Self:
        assert entry.message_code == 329, "Entry must be a link went down event"
        match = LINK_WENT_DOWN_MESSAGE_PATTERN.match(entry.message)
        if match is None:
            raise ValueError(f"Message does not match expected pattern: {entry.message}")
        ports = [
            PortId(switch=match.group("switch"), port=match.group("port")),
            PortId(switch=match.group("peer_switch"), port=match.group("peer_port")),
        ]
        ports = standardize_ports(ports)
        return cls(timestamp=entry.timestamp, ports=tuple(sorted(ports, key=str)), original_line=entry.original_line)

    @classmethod
    def build_from_port_considered_unhealthy_entry(cls, entry: Entry) -> Self:
        assert entry.message_code == 702, "Entry must be a port is considered unhealthy event"
        ports = [PortId.build_from_device_description(entry.device_description)]
        match = PORT_IS_CONSIDERED_UNHEALTHY_MESSAGE_PATTERN.match(entry.message)
        if match is not None:
            ports.append(PortId(switch=match.group("switch"), port=match.group("port")))
        ports = standardize_ports(ports)
        return cls(timestamp=entry.timestamp, ports=tuple(sorted(ports, key=str)), original_line=entry.original_line)

    @classmethod
    def build_from_symbol_bit_error_rate_entry(cls, entry: Entry) -> Self:
        assert entry.message_code in (917, 918), "Entry must be a symbol bit error rate event"
        port_id = PortId.build_from_device_description(entry.device_description)
        return cls(timestamp=entry.timestamp, ports=(port_id,), original_line=entry.original_line)


@attr.s(auto_attribs=True, frozen=True)
class PortId:
    switch: str
    port: str

    def __str__(self) -> str:
        return f"{self.switch} {self.port}"

    def counterpart(self) -> Self:
        prefix, port_idx_str = self.port.rsplit("/", 1)
        new_port_idx = 1 if int(port_idx_str) == 2 else 2
        return type(self)(switch=self.switch, port=f"{prefix}/{new_port_idx}")

    @classmethod
    def build_from_device_description(cls, device_description: str) -> Self:
        try:
            _default, switch, port = device_description.split(" / ", 2)
        except ValueError as e:
            raise ValueError(f"Device description does not match expected format: {device_description}") from e
        return cls(switch.removeprefix("Switch: "), port.removeprefix("1/"))

    @classmethod
    def build_from_informal_description(cls, informal_description: str) -> Self:
        """Sometimes we get back descriptions of ports that look like this:
        E17-L-U43 17/1
        """
        abbreviated_switch, port = informal_description.split(" ", 1)
        # Find the full switch name.
        for port_id in get_ports_by_port_id().keys():
            if port_id.switch.endswith(abbreviated_switch):
                return cls(port_id.switch, port)
        raise ValueError(f"Could not find switch with abbreviated name: {abbreviated_switch}")


def read_entries(prune_consecutive=True) -> Tuple[Entry, ...]:
    result = []
    with download_event_log_locally() as event_path_filename:
        with open(event_path_filename, "r") as f:
            for line in f.readlines():
                try:
                    result.append(Entry.build_from_line(line))
                except Exception:
                    print(f"Failed to parse line: {line}", file=ERROR_FILE)
    if prune_consecutive:
        result = prune_many_consecutive_entries(result)
    return tuple(result)


BASE_CUTOFF_TIMESTAMP: Final[datetime.datetime] = datetime.datetime(
    year=2024, month=6, day=20, hour=0, minute=0, second=0, tzinfo=ZoneInfo("America/Los_Angeles")
)


def should_include(event: PortRelatedEvent, entry: Entry) -> bool:
    if "quincy" in entry.message:
        return False
    if any("Computer" in port_id.switch for port_id in event.ports):
        print(f"Skipping event for computer: {entry.original_line.strip()}", file=ERROR_FILE)
        return False
    if any(read_peer_port_mapping().get(port_id) is None for port_id in event.ports):
        print(f"Not skipping event for port without peer: {entry.original_line.strip()}")
    return True


MESSAGE_CODES_TO_HANDLE: Final[Tuple[int, ...]] = (
    110,  # Symbol error counter rate
    112,  # Link downed
    113,  # Port receive errors
    115,  # Receive switch relay errors
    116,  # Transmit discards
    329,  # Link went down
    702,  # Port is considered unhealthy
    917,  # Symbol bit error rate
    918,  # Symbol bit error rate warning
)


MESSAGE_CODES_TO_IGNORE: Final[Tuple[int, ...]] = (
    64,  # GID address out of service
    65,  # GID address in service
    66,  # MCast group created
    67,  # MCast group deleted
    328,  # Link went up
    331,  # Node is down
    332,  # Node is up
    336,  # Port action disable succeeded
    395,  # Action get_cables_info started
    517,  # Fabric health report
    527,  # UFM CPU usage
    603,  # UFM non-critical event suppression
    604,  # Fabric analysis report succeeded
    908,  # Switch up
    1500,  # New cable detected
    1502,  # Cable detected in a new location
    # These should be handled at some point.
    394,  # Switch critical failure
    907,  # Switch down
    920,  # Cable Low Temperature Alarm reported
    1503,  # Duplicate cable detected
)


def latest_port_related_events(entries: Tuple[Entry, ...]) -> Dict[Tuple[PortId, ...], PortRelatedEvent]:
    seen_unexpected_message_codes: Set[int] = set()
    latest_events_by_ports: Dict[Tuple[PortId, ...], PortRelatedEvent] = {}
    for entry in entries:
        if entry.timestamp < BASE_CUTOFF_TIMESTAMP:
            print(f"Skipping entry before cutoff: {entry.original_line.strip()}", file=ERROR_FILE)
            continue
        if "Aggregation Node" in entry.message or "Aggregation Node" in entry.device_description:
            print(f"Skipping aggregation node event: {entry.original_line.strip()}", file=ERROR_FILE)
            continue
        if entry.message_code in MESSAGE_CODES_TO_IGNORE:
            continue
        elif entry.message_code in MESSAGE_CODES_TO_HANDLE:
            if entry.message_code == 329:
                if "Computer" in entry.message or "Aggregation Node" in entry.message:
                    continue
                event = PortRelatedEvent.build_from_link_went_down_entry(entry)
            elif entry.message_code in THRESHOLD_EXCEEDED_CODES:
                event = PortRelatedEvent.build_from_threshold_exceeded_entry(entry)
            elif entry.message_code == 702:
                if "MANUAL" in entry.message:
                    continue
                event = PortRelatedEvent.build_from_port_considered_unhealthy_entry(entry)
            elif entry.message_code in (917, 918):
                event = PortRelatedEvent.build_from_symbol_bit_error_rate_entry(entry)
            else:
                raise ValueError(f"Unexpected message code: {entry.message_code}")
            if not should_include(event, entry):
                continue
            existing = latest_events_by_ports.get(event.ports)
            if existing is None or entry.timestamp > existing.timestamp:
                latest_events_by_ports[event.ports] = event
        else:
            if entry.message_code not in seen_unexpected_message_codes:
                print(f"Unexpected message code: {entry.message_code}: {entry.original_line.strip()}")
                seen_unexpected_message_codes.add(entry.message_code)
    return latest_events_by_ports


# These numbers are picked somewhat arbitrarily.
MAX_EVENTS_IN_TIMEFRAME = 50
TIMEFRAME_SECONDS = 1


def prune_many_consecutive_entries(entries: List[Entry, ...]) -> List[Entry, ...]:
    """
    Some big events in ufm (such as a reboot) cause a lot of events in quick succession.
    We expect most of these events to be noise and disregard them.
    """
    entries = sorted(entries, key=lambda entry: entry.timestamp)
    prev_entries: List[Entry] = [entries[0]]
    prev_start_time: datetime = entries[0].timestamp
    final_entries = []
    for entry in entries:
        entry_time = entry.timestamp
        if entry_time - prev_start_time < datetime.timedelta(seconds=TIMEFRAME_SECONDS):
            prev_entries.append(entry)
            prev_start_time = entry_time
        else:
            if len(prev_entries) < MAX_EVENTS_IN_TIMEFRAME:
                final_entries.extend(prev_entries)
            else:
                print(f"Skipping {len(prev_entries)} entries with start of {prev_start_time}")
            prev_entries = [entry]
            prev_start_time = entry_time
    print(f"Pruned {len(entries) - len(final_entries)} entries, leaving {len(final_entries)} entries.")
    return final_entries


def get_actions_from_logs(entries: Tuple[Entry, ...]) -> Tuple[Action, ...]:
    latest_events = latest_port_related_events(entries)
    results = {}
    for ports, event in latest_events.items():
        for port in ports:
            if port in results:
                continue
            results[port] = DisablePortAction(
                port=port,
                cause=event,
            )
    return tuple(sorted(results.values(), key=str))


def write_action_file(actions: Iterable[Action], filename: str) -> None:
    with open(filename, "w") as f:
        for action in actions:
            f.write(f"{action}\n")


def process_logs(output_filename: str) -> None:
    entries = read_entries(prune_consecutive=True)
    actions = get_actions_from_logs(entries)
    write_action_file(actions, output_filename)


UFM_EVENT_LOG_PATH: Final[str] = "/opt/ufm/log/event.log"


@contextmanager
def download_event_log_locally() -> Iterator[Path]:
    with tempfile.TemporaryDirectory() as temp_dir:
        event_log_path = Path(temp_dir) / "event.log"
        subprocess.run(
            (
                "scp",
                f"host@{UFM_LOCAL_IP}:{UFM_EVENT_LOG_PATH}",
                str(event_log_path),
            ),
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        yield event_log_path


def convert(s: str) -> PortId:
    s = s.replace(":", "-", 1)
    switch, port = s.split(":")
    if port.startswith("0"):
        port = port[1:]
    return PortId(switch, port)


@functools.cache
def read_peer_port_mapping() -> Dict[PortId, PortId]:
    this_path = Path(__file__)
    sid_txt = this_path.parent.parent / "portcheck/inputs/sid.txt"
    mapping = {}

    with open(sid_txt, "r") as f:
        for line in f:
            if "IB" in line:
                continue
            left, right = line.strip().split(" ")
            left_pid = convert(left)
            right_pid = convert(right)
            mapping[left_pid] = right_pid
            mapping[right_pid] = left_pid
    return mapping


def get_portid(switch: str, num: int) -> PortId:
    port_str = f"{(num + 1) // 2}/{num % 2 if num % 2 == 1 else 2}"
    portid = PortId(switch, port_str)
    return portid


def get_bad_ports_and_peers_from_counters(bad_ports_file: Path, iblinkinfo_file: Path) -> None:
    """
    Find bad ports based on their error counters as produced by ibpc and counters.sh on the ufm box
    """
    previously_disabled_ports = set(parse_bad_ports(iblinkinfo_file))
    with open(bad_ports_file) as f:
        lines = f.readlines()
        for line in lines[1:]:
            guid, lid, device, switch, port_num, active, link_down, tb_sent, tb_recv, stable = line.split()
            port = get_portid(switch, int(port_num))
            if port in previously_disabled_ports:
                print(f"Port {port} is already disabled")
                continue
            peer_port = read_peer_port_mapping().get(port)
            if peer_port is None:
                # In our networking setup, T2 switches can directly connect to hosts, so it doesn't make sense
                # to disable the "peer" port in this case.
                assert (
                    port.switch.startswith("T2") and int(port.port.split("/")[0]) <= 16
                ), f"Peer port not found for {port}"
                continue
            previously_disabled_ports.add(port)
            previously_disabled_ports.add(peer_port)


def parse_bad_ports(filename: Path) -> List[PortId]:
    """
    Parses a file that contains the output directly from iblinkinfo
    """
    pattern = re.compile(r"(\d+)\[\s*\]")
    ports = []
    current_switch = None
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("Switch"):
                current_switch = line.split(" ")[-1].strip().replace(":", "")
            if "Down/ Polling" in line or "Initialize" in line:
                number_group = re.search(pattern, line)
                number = int(number_group.group(1))
                ports.append(get_portid(current_switch, number))
    return ports


def get_ports_with_bad_states_iblinkinfo(
    infile: Path, outfile: Path, only_bad_ports: bool = False
) -> None:
    """
    Finds ports and peers of ports that are in a bad state (Polling or Initialized).
    If only_bad_ports is True, then the infile should have a port and number on each line
    Otherwise, the infile should be the output of iblinkinfo
    """
    if only_bad_ports:
        with open(infile, "r") as f:
            port_strs = [line.strip().split() for line in f.readlines()]
            ports = [get_portid(switch, int(num)) for switch, num in port_strs]
    else:
        ports = parse_bad_ports(infile)
    date_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    to_disable_ports = set()
    with open(outfile, "w") as f:
        print(f"Found total of {len(ports)} ports in bad state")
        # For each port in a bad state, find its peer port
        for port in ports:
            peer_port = read_peer_port_mapping().get(port)
            if peer_port is None:
                # Skip the peer port if it's a T2 connected directly to a host
                assert (
                    port.switch.startswith("T2") and int(port.port.split("/")[0]) <= 16
                ), f"Peer port not found for {port}"
                continue
            to_disable_ports.add(port)
            to_disable_ports.add(peer_port)
            f.write(f"{port}\n")
            f.write(f"{peer_port}\n")
        f.write(f"Found {len(to_disable_ports)} ports and peers of bad ports on {date_str}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suppress-errors", action="store_true")
    command_parsers = parser.add_subparsers(dest="command")

    process_logs_parser = command_parsers.add_parser("process-logs")
    process_logs_parser.add_argument("output_filename")
    process_logs_parser.set_defaults(func=process_logs)

    disable_from_file_parser = command_parsers.add_parser("get-bad-from-iblinkinfo")
    disable_from_file_parser.add_argument("--infile")
    disable_from_file_parser.add_argument("--outfile")
    disable_from_file_parser.add_argument("--only-bad-ports", action="store_true")
    disable_from_file_parser.set_defaults(func=get_ports_with_bad_states_iblinkinfo)

    disable_from_counters_parser = command_parsers.add_parser("get-bad-from-counters")
    disable_from_counters_parser.add_argument("--bad-ports-file")
    disable_from_counters_parser.add_argument("--iblinkinfo-file")
    disable_from_counters_parser.set_defaults(func=get_bad_ports_and_peers_from_counters)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.suppress_errors:
        global ERROR_FILE
        ERROR_FILE = open(os.devnull, "w")

    args_to_ignore = ("suppress_errors", "func", "command")
    args.func(**{k: v for k, v in vars(args).items() if k not in args_to_ignore})


if __name__ == "__main__":
    main()
