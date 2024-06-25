"""
Usage: python run_health_checks.py <nodes> where nodes is a comma separated list of nodes
or provide a list of nodes as a list of strings in the config
"""


import sys
import time
from multiprocessing import Process
from multiprocessing import Queue
import os
import json

from health_checks import ComputeHostHealth, HealthCheck, get_health_check_from_str
from health_checks import HealthCheckCommandError
from health_checks import ALL_HEALTH_CHECKS
from health_checks import outcome_to_health_check_result
from utils.commands import SSHConnectionData
from utils.commands import SUBPROCESS_STOPPED_BY_REQUEST_EXIT_CODE
from utils.commands import run_remote_command

EXPECTED_CONFIG_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.json')
with open(EXPECTED_CONFIG_FILE, 'r') as f:
    EXPECTED_CONFIG = json.load(f)

def run_health_check(node: str, health_check: HealthCheck, queue: Queue, timeout_sec: int = 100) -> None:
    connection = SSHConnectionData(ip=node, port=EXPECTED_CONFIG["node_info"]["port"], user=EXPECTED_CONFIG["node_info"]["user"])
    command = health_check.create_command()
    try:
        result = run_remote_command(
            machine_ssh_command=connection.get_ssh_command(),
            remote_command=command,
            is_checked=True,
            timeout_sec=timeout_sec,
        )
    except Exception as e:
        # If the exception does not have a returncode, we use a known sentinel.
        NO_RETURN_CODE = -1
        returncode = getattr(e, "returncode", NO_RETURN_CODE)
        if returncode == SUBPROCESS_STOPPED_BY_REQUEST_EXIT_CODE:
            print("Health check timed out.")
        else:
            print(f"Health check failed with return code: {returncode}")
        queue.put(
            (connection.ip, HealthCheckCommandError(message=str(e), returncode=returncode, cause=str(returncode)))
        )
        return
    outcome = health_check.validate_result(result.output, result.returncode)
    queue.put((connection.ip, outcome))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        nodes = sys.argv[1]
        node_list = nodes.split(",")
    elif "node_info" in EXPECTED_CONFIG and "nodes" in EXPECTED_CONFIG["node_info"]:
        node_list = EXPECTED_CONFIG["node_info"]["nodes"]
        if len(node_list) == 0:
            raise ValueError("No nodes provided, please provide either through command line or config.")
    else:
        raise ValueError("No nodes provided, please provide either through command line or config.")
    print("Running health checks on nodes: ", node_list)
    health_check = ALL_HEALTH_CHECKS
    if "leaf_health_checks" in EXPECTED_CONFIG:
        leaf_health_checks_str = EXPECTED_CONFIG["leaf_health_checks"]
        if len(leaf_health_checks_str) == 0:
            print("No health checks provided, running all health checks")
        else:
            health_check = get_health_check_from_str(",".join(EXPECTED_CONFIG["leaf_health_checks"]))
            print("Running health checks: ", health_check)
            if health_check is None:
                print("Couldn't find health checks", EXPECTED_CONFIG["leaf_health_checks"])
                health_check = ALL_HEALTH_CHECKS
    processes = []
    queue = Queue()
    health_check_timeout = 100
    for node in node_list:
        p = Process(
            target=run_health_check,
            args=(node, health_check, queue, health_check_timeout),
        )
        processes.append(p)
        p.start()
    start_time = time.time()
    results = []
    while len(results) < len(processes) or not queue.empty():
        results.append(queue.get())

    for p in processes:
        p.join()

    health_check_results = {
        ComputeHostHealth.OK: [],
        ComputeHostHealth.UNHEALTHY: [],
        ComputeHostHealth.CRITICAL: [],
        ComputeHostHealth.UNKNOWN: [],
    }

    for result in results:
        node, outcome = result
        health_check_results[outcome_to_health_check_result(outcome)].append((node, outcome))

    for health, node_outcomes in health_check_results.items():
        if len(node_outcomes) > 0:
            print(f"Health {health}")
        if health == ComputeHostHealth.OK:
            continue
        for node, outcome in node_outcomes:
            message = outcome.message.replace("\n", "\n\t")
            remediation = outcome.suggested_remediation.replace("\n", "\n\t")
            print(f"{node}: \n{message}\n{remediation}")

    print("\n\n----------------------------SUMMARY:----------------------------\n")

    for health, node_outcomes in health_check_results.items():
        nodes = [node_outcome[0] for node_outcome in node_outcomes]
        print(f"Nodes with {health}: {nodes}")
