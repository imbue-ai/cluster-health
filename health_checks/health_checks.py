"""A library of health checks for hosts.
"""

import csv
import datetime
import json
import os
import re
from abc import ABC
from abc import abstractmethod
from enum import Enum
from typing import Container
from typing import Dict
from typing import Final
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
from uuid import uuid4

import attr

from dmesg_whitelist import WHITELISTED_MESSAGES
from dmesg_whitelist import WHITELISTED_MESSAGE_RANGES
from dmesg_whitelist import WHITELISTED_NVSWITCH_SXID_ERRORS
from dmesg_whitelist import WHITELISTED_REGEX_STR
from utils.commands import remove_none


EXPECTED_CONFIG_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.json')
with open(EXPECTED_CONFIG_FILE, 'r') as f:
    EXPECTED_CONFIG = json.load(f)


class ComputeHostHealth(Enum):
    """The health of a compute host.

    Details:
    - If health checks are not implemented, the health will always be UNKNOWN.
    - Hosts that are labeled as "UNHEALTHY" may still be used for allocation.
    - Hosts that are labeled as "CRITICAL" should not be used for allocation and we should produce an alert.
    """
    OK = "OK"  # lintignore
    """Host is known to be healthy."""
    CRITICAL = "CRITICAL"  # lintignore
    """Host is not suitable for allocation."""
    UNHEALTHY = "UNHEALTHY"  # lintignore
    """Host has some undesirable state that should be investigated, but may still be used for allocation."""
    UNKNOWN = "UNKNOWN"  # lintignore
    """Either provider has not been set up yet, or we failed to run health checks to completion."""


@attr.s(auto_attribs=True, frozen=True)
class HealthyResult:
    """
    Indicates a successful health check.
    """

    message: str = "OK"
    suggested_remediation: Optional[str] = None

    def create_fix_command(self, ip: str, user: str, port:int) -> None:
        return None


@attr.s(auto_attribs=True, frozen=True)
class HealthCheckSilencedWarning:
    """
    Host did not pass the health check, but the issue is currently disabled.

    Attributes:
        message: Human-readable string.
        suggested_remediation: Human-readable string.
    """

    message: str
    suggested_remediation: Optional[str] = None

    def create_fix_command(self, ip: str, user: str, port:int) -> Optional[str]:
        return None

    def display(self, indentation: int = 0) -> str:
        """Produce a string with a human-readable summary of the health check error that has occurred.

        Accepts an optional number of spaces by which to indent each line of the message.
        """
        message = type(self).__name__ + ": " + self.message + "\n"
        if self.suggested_remediation is not None:
            message += f"Suggested remediation: {self.suggested_remediation}"

        message = "\n".join((indentation * " ") + line for line in message.splitlines())
        return message


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class CompoundHealthCheckSilencedWarning(HealthCheckSilencedWarning):
    """
    A silenced error that is composed of multiple silenced errors
    """

    @classmethod
    def build_from_list(cls, silenced_warnings: Sequence[HealthCheckSilencedWarning]) -> HealthCheckSilencedWarning:
        """
        Builds a compound silenced error from a list of sub-errors in a human-readable way,
        or just returns the singleton warning passed in.
        """
        if len(silenced_warnings) == 1:
            return silenced_warnings[0]

        message = "\n".join(
            (
                "Several issues were found:",
                *(f"{idx}: {obj.__class__.__name__}: {obj.message}" for idx, obj in enumerate(silenced_warnings)),
            )
        )
        suggested_remediation = "\n".join(
            (
                "Several issues were found, and the suggested remediations are:",
                *(
                    f"{idx}: {obj.__class__.__name__}: {obj.suggested_remediation or 'Unknown'}"
                    for idx, obj in enumerate(silenced_warnings)
                ),
            )
        )

        return cls(
            message=message,
            suggested_remediation=suggested_remediation,
        )


@attr.s(auto_attribs=True, frozen=True)
class HealthCheckWarning:
    """
    Host did not pass the health check, but the issue is temporarily acceptable, or somehow not critical.

    Attributes:
        message: Human-readable string.
        suggested_remediation: Human-readable string.
    """

    message: str
    suggested_remediation: Optional[str] = None

    def create_fix_command(self, ip: str, user: str, port:int) -> Optional[str]:
        return None

    def display(self, indentation: int = 0) -> str:
        """Produce a string with a human-readable summary of the health check error that has occurred.

        Accepts an optional number of spaces by which to indent each line of the message.
        """
        message = type(self).__name__ + ": " + self.message + "\n"
        if self.suggested_remediation is not None:
            message += f"Suggested remediation: {self.suggested_remediation}"

        message = "\n".join((indentation * " ") + line for line in message.splitlines())
        return message


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class CompoundHealthCheckWarning(HealthCheckWarning):
    """
    A warning that is composed of multiple warnings
    """

    @classmethod
    def build_from_list(
        cls, warnings: Sequence[HealthCheckWarning], silenced_warnings: Sequence[HealthCheckSilencedWarning] = ()
    ) -> HealthCheckWarning:
        """
        Builds a compound warning from a list of sub-warnings in a human-readable way,
        or just returns the singleton warning passed in.
        """
        seq: Tuple[HealthCheckOutcome] = (
            *warnings,
            *silenced_warnings,
        )  # type: ignore
        if len(seq) == 1:
            return warnings[0]

        message = "\n".join(
            (
                "Several issues were found:",
                *(f"{idx}: {obj.__class__.__name__}: {obj.message}" for idx, obj in enumerate(seq)),
            )
        )
        suggested_remediation = "\n".join(
            (
                "Several issues were found, and the suggested remediations are:",
                *(
                    f"{idx}: {obj.__class__.__name__}: {obj.suggested_remediation or 'Unknown'}"
                    for idx, obj in enumerate(seq)
                ),
            )
        )

        return cls(
            message=message,
            suggested_remediation=suggested_remediation,
        )


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class HealthCheckError:
    """
    Information about a health check error.
    """

    message: str
    suggested_remediation: str
    cause: Optional[str] = None

    def create_fix_command(self, ip: str, user: str, port:int) -> Optional[str]:
        return None

    def display(self, indentation: int = 0) -> str:
        """Produce a string with a human-readable summary of the health check error that has occurred.

        Accepts an optional number of spaces by which to indent each line of the message.
        """
        message = type(self).__name__ + ": " + self.message + "\n"
        message += f"Suggested remediation: {self.suggested_remediation}"
        if self.cause is not None:
            message += "\nCause: " + self.cause

        message = "\n".join((indentation * " ") + line for line in message.splitlines())
        return message


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class CompoundHealthCheckError(HealthCheckError):
    """
    An error that is composed of at least 1 error and also possibly some warnings.
    """

    @classmethod
    def build_from_list(
        cls,
        errors: Sequence[HealthCheckError],
        warnings: Sequence[HealthCheckWarning] = (),
        silenced_warnings: Sequence[HealthCheckSilencedWarning] = (),
    ) -> HealthCheckError:
        """
        Builds a compound error from a list of sub-errors + warnings in a human-readable way,
        or just returns the singleton error passed in.
        """
        seq: Tuple[HealthCheckOutcome] = (
            *errors,
            *warnings,
            *silenced_warnings,
        )  # type: ignore
        if len(seq) == 1:
            return errors[0]

        message = "\n".join(
            (
                "Several issues were found:",
                *(f"{idx}: {obj.__class__.__name__}: {obj.message}" for idx, obj in enumerate(seq)),
            )
        )
        suggested_remediation = "\n".join(
            (
                "Several issues were found, and the suggested remediations are:",
                *(
                    f"{idx}: {obj.__class__.__name__}: {obj.suggested_remediation or 'Unknown'}"
                    for idx, obj in enumerate(seq)
                ),
            )
        )

        return cls(
            message=message,
            suggested_remediation=suggested_remediation,
            cause=None,
        )


@attr.s(auto_attribs=True, frozen=True)
class HealthCheckIncomplete:
    """
    The result of a health check that failed to complete successfully.

    Use this for health checks which have external dependencies (e.g. network) which may be flaky
    but independent of the state of the host, or other temporary infrastructure failures.
    """

    message: str
    suggested_remediation: Optional[str] = None

    def create_fix_command(self, ip: str, user: str, port:int) -> None:
        return None

    def display(self, indentation: int = 0) -> str:
        """Produce a string with a human-readable summary of the health check error that has occurred.

        Accepts an optional number of spaces by which to indent each line of the message.
        """
        message = type(self).__name__ + ": " + self.message + "\n"
        if self.suggested_remediation is not None:
            message += f"Suggested remediation: {self.suggested_remediation}"

        message = "\n".join((indentation * " ") + line for line in message.splitlines())
        return message


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class CompoundHealthCheckIncomplete(HealthCheckIncomplete):
    """
    A result that represents a health check that failed to complete successfully,
    plus possibly other incomplete health checks, or health errors or warnings.
    """

    @classmethod
    def build_from_list(
        cls,
        incompletes: Sequence[HealthCheckIncomplete],
        errors: Sequence[HealthCheckError] = (),
        warnings: Sequence[HealthCheckWarning] = (),
        silenced_warnings: Sequence[HealthCheckSilencedWarning] = (),
    ) -> HealthCheckIncomplete:
        """
        Builds a compound incomplete_result from a list of incomplete results
        + possibly any errors or warnings.
        If there is only one incomplete result and no errors/warnings, it is instead simply returned.
        """
        seq: Tuple[HealthCheckOutcome] = (*incompletes, *errors, *warnings, *silenced_warnings)  # type: ignore
        if len(seq) == 1:
            return incompletes[0]

        message = "\n".join(
            (
                "Several issues were found:",
                *(f"{idx}: {obj.__class__.__name__}: {obj.message}" for idx, obj in enumerate(seq)),
            )
        )
        suggested_remediation = "\n".join(
            (
                "Several issues were found, and the suggested remediations are:",
                *(
                    f"{idx}: {obj.__class__.__name__}: {obj.suggested_remediation or 'Unknown'}"
                    for idx, obj in enumerate(seq)
                ),
            )
        )

        return cls(
            message=message,
            suggested_remediation=suggested_remediation,
        )


"""
A type alias for the possible outcomes of a health check:
 - Healthy (passed the health check)
 - Error (failed the health check)
 - Warning (degraded)
 - Incomplete (the health check failed to give enough information to determine the health)
 - SilencedWarning (the health check is currently disabled)
"""
HealthCheckOutcome = Union[
    HealthyResult, HealthCheckError, HealthCheckWarning, HealthCheckIncomplete, HealthCheckSilencedWarning
]


def outcome_to_health_check_result(outcome: HealthCheckOutcome) -> ComputeHostHealth:
    if isinstance(outcome, HealthyResult):
        return ComputeHostHealth.OK
    elif isinstance(outcome, HealthCheckSilencedWarning):
        return ComputeHostHealth.OK
    elif isinstance(outcome, HealthCheckWarning):
        return ComputeHostHealth.UNHEALTHY
    elif isinstance(outcome, HealthCheckError):
        return ComputeHostHealth.CRITICAL
    elif isinstance(outcome, HealthCheckCommandError):
        return ComputeHostHealth.UNKNOWN
    else:
        return ComputeHostHealth.UNKNOWN
HEALTH_CHECK_FIX_DIR = "health_checks/health_check_fixes/"


class HealthCheck(ABC):
    """The core interface for health checks.

    A health check consists of a bash command that is run on the host and a validation
    function that parses the output of the command and returns an error if the output
    indicates that the host may be unhealthy.
    """

    @abstractmethod
    def create_command(self) -> str:
        """Create the command to run."""

    @abstractmethod
    def validate_result(self, output: str, returncode: int) -> HealthCheckOutcome:
        """Parse the output of the command."""


class LeafHealthCheck(HealthCheck):
    """A health check that is a leaf in the health check tree.

    A leaf health check is a health check that is not composed of other health checks.

    This is mostly a marker to avoid accidentally nesting compound health checks.
    """


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class HealthCheckCommandError(HealthCheckIncomplete):
    """Returned when there is a problem with the health check command."""

    returncode: int
    cause: Optional[str]
    suggested_remediation: str = "Error or timeout launching health checks. Maybe the host is not reachable?"


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class HealthCheckCompoundCommandError(HealthCheckCommandError):
    """Returned when there is a problem with the compound health check command."""

    suggested_remediation: str = "Error running health checks in parallel. Maybe the host is not reachable?"


_COMPOUND_HEALTH_CHECK_END_DELIMITER: Final[str] = "COMPOUND_HEALTH_CHECK_END_DELIMITER"
_RETURNCODE_PREFIX: Final[str] = "RETURNCODE:"


@attr.s(auto_attribs=True, frozen=True)
class CompoundHealthCheck(HealthCheck):
    """
    A health check that is composed of multiple health checks.

    The health check commands are run in parallel, and the output of each command is
    validated independently. If any of the health checks fail, the compound health check
    fails.
    """

    health_checks: Tuple[HealthCheck, ...]

    def _get_leafhealth_checks(self) -> Iterable[LeafHealthCheck]:
        for health_check in self.health_checks:
            if isinstance(health_check, LeafHealthCheck):
                yield health_check
            elif isinstance(health_check, CompoundHealthCheck):
                yield from health_check._get_leafhealth_checks()
            else:
                raise ValueError(f"Unexpected health check type: {type(health_check)}")

    def create_command(self) -> str:
        """Creates a bash command that runs the health checks in parallel,
        but produces the output of each health check in order, in a predictable format.

        The output of each health check is delimited by a start and end delimiter such that compound
        health checks can be nested.

        The mechanism that we use to run the health checks in parallel is to create a temporary file for
        each health check, and then run the health check in a subshell in the background, redirecting
        the output of the health check to the file. We then read from each file in order
        and print the output of each health check, delimited by the start and end delimiters.
        """
        tempfile_variable_names = [f"tmpfile{str(uuid4()).replace('-', '_')}" for _ in self._get_leafhealth_checks()]
        commands: List[str] = []
        for idx, health_check in enumerate(self._get_leafhealth_checks()):
            # Execute in a subshell in the background.
            commands.extend(
                [
                    f"{tempfile_variable_names[idx]}=$(mktemp)",
                    f"({health_check.create_command()}) > ${tempfile_variable_names[idx]} 2>&1 &",
                    f"pid{idx}=$!",
                ]
            )

        for idx, _ in enumerate(self._get_leafhealth_checks()):
            commands.extend(
                [
                    f"wait $pid{idx}",
                    f'echo "{_RETURNCODE_PREFIX}$?"',
                    f"cat ${tempfile_variable_names[idx]}",
                    f"echo {_COMPOUND_HEALTH_CHECK_END_DELIMITER}",
                    f"rm ${tempfile_variable_names[idx]}",
                ]
            )

        return "\n".join(commands)

    def get_health_check_map(self, output: str, returncode: int) -> Dict[LeafHealthCheck, HealthCheckOutcome]:
        health_check_map: Dict[LeafHealthCheck, HealthCheckOutcome] = dict()
        if len(list(self._get_leafhealth_checks())) == 0:
            return health_check_map
        outputs = iter(output.splitlines())
        command_outputs: List[str] = []
        command_returncodes: List[int] = []
        current_output: List[str] = []
        for line in outputs:
            if line.startswith(_COMPOUND_HEALTH_CHECK_END_DELIMITER):
                command_outputs.append("\n".join(current_output))
                current_output = []
                continue
            elif line.startswith(_RETURNCODE_PREFIX):
                returncode_part = line.removeprefix(_RETURNCODE_PREFIX)
                command_returncodes.append(int(returncode_part))
                continue
            current_output.append(line)

        for health_check, command_output, returncode in zip(
            self._get_leafhealth_checks(), command_outputs, command_returncodes
        ):
            health_check_result = health_check.validate_result(command_output, returncode)
            health_check_map[health_check] = health_check_result
        return health_check_map

    def validate_result(self, output: str, returncode: int) -> HealthCheckOutcome:
        if returncode != 0:
            return HealthCheckCompoundCommandError(
                message=f"The compound health check command failed with output: {output}",
                returncode=returncode,
                cause=None,
            )

        health_check_outcomes = list(self.get_health_check_map(output, returncode).values())
        if len(health_check_outcomes) == 0:
            return HealthyResult()
        return make_compound_error(health_check_outcomes)


def make_compound_error(outcomes: List[HealthCheckOutcome]) -> HealthCheckOutcome:
    """
    Merges a mixed list of errors/warnings/oks/incompletes into a single compound result.

    Incomplete results that indicate something wrong with health check code override the other types.
    Then come the errors if any, then warnings if any, then only if everything in the list is OK can we
    declare the compound result is OK.
    """

    incompletes: List[HealthCheckIncomplete] = []
    errors: List[HealthCheckError] = []
    warnings: List[HealthCheckWarning] = []
    silenced_warnings: List[HealthCheckSilencedWarning] = []

    for health_check_result in outcomes:
        if isinstance(health_check_result, HealthCheckIncomplete):
            incompletes.append(health_check_result)
        if isinstance(health_check_result, HealthCheckError):
            errors.append(health_check_result)
        if isinstance(health_check_result, HealthCheckWarning):
            warnings.append(health_check_result)
        if isinstance(health_check_result, HealthCheckSilencedWarning):
            silenced_warnings.append(health_check_result)

    if incompletes:
        return CompoundHealthCheckIncomplete.build_from_list(incompletes, errors, warnings, silenced_warnings)
    elif errors:
        return CompoundHealthCheckError.build_from_list(errors, warnings, silenced_warnings)
    elif warnings:
        return CompoundHealthCheckWarning.build_from_list(warnings, silenced_warnings)
    elif silenced_warnings:
        return CompoundHealthCheckSilencedWarning.build_from_list(silenced_warnings)
    else:
        return HealthyResult()


def get_hca_cards() -> List[str]:
   return EXPECTED_CONFIG["ib_hcas"].keys()



@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class GpuHealthCheckError(HealthCheckError):
    """Raised when there is a problem with the GPUs."""

    suggested_remediation: str = "\n".join(
        (
            "We have identified a hardware error on one or more GPUs.",
            "Some times a restart will fix the issue.",
            "If it persists this host should not be used and we should communicate with the data center to determine next steps.",
        )
    )

    def create_fix_command(self, ip: str, user: str, port:int) -> str:
        return f"bash {HEALTH_CHECK_FIX_DIR}fix_nvidia_smi.sh {ip} {user} {port}"


@attr.s(auto_attribs=True, frozen=True)
class NvidiaSmiHealthCheck(LeafHealthCheck):
    def create_command(self) -> str:
        return "nvidia-smi --query-gpu=index,ecc.errors.uncorrected.volatile.total,ecc.mode.current,ecc.mode.pending --format=csv,noheader,nounits"

    def validate_result(self, output: str, returncode: int) -> HealthCheckOutcome:
        if returncode != 0:
            return GpuHealthCheckError(message=f"The `nvidia-smi` command failed with return code {returncode}.")
        csv_reader = csv.reader(output.splitlines())
        for row in csv_reader:
            if len(row) != 4:
                return GpuHealthCheckError(message=f"Expected 4 columns, but found {len(row)}: {row}")
            try:
                gpu_index, ecc_errors, ecc_mode_current, ecc_mode_pending = (
                    int(row[0]),
                    int(row[1]),
                    row[2].strip(),
                    row[3].strip(),
                )
            except ValueError as e:
                return GpuHealthCheckError(message=f"Failed to parse row: {row}", cause=repr(e))
            if ecc_mode_current != "Enabled" or ecc_mode_pending != "Enabled":
                return GpuHealthCheckError(
                    message=f"Expected ECC to be enabled, but found {ecc_mode_current},pending:{ecc_mode_pending} on GPU {gpu_index}."
                )
            if ecc_errors != 0:
                return GpuHealthCheckError(
                    message=f"Expected 0 ECC errors, but found {ecc_errors} on GPU {gpu_index}."
                )
        return HealthyResult()


NVIDIA_SMI_HEALTH_CHECK: Final[LeafHealthCheck] = NvidiaSmiHealthCheck()


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class NvidiaFabricHealthCheckError(HealthCheckError):
    """Raised when there is a problem with NVIDIA Fabric Manager."""

    suggested_remediation: str = "\n".join(
        (
            "We have identified a hardware error with NVIDIA Fabric Manager.",
            "We can try to restart the service to fix the issue.",
            "This host should not be used and we should communicate with the data center to determine next steps.",
        )
    )

    def create_fix_command(self, ip: str, user: str, port:int) -> str:
        return f"bash {HEALTH_CHECK_FIX_DIR}fix_nvidia_fabric_manager.sh {ip} {user} {port}"


@attr.s(auto_attribs=True, frozen=True)
class NvidiaFabricManagerHealthCheck(LeafHealthCheck):
    def create_command(self) -> str:
        return "systemctl is-active nvidia-fabricmanager"

    def validate_result(self, output: str, returncode: int) -> HealthCheckOutcome:
        if returncode != 0:
            return NvidiaFabricHealthCheckError(
                message=f"The `systemctl` command failed with return code {returncode}."
            )
        if output.strip() != "active":
            return NvidiaFabricHealthCheckError(
                message=f"nvidia-fabricmanager is not active, returned {output.strip()}."
            )
        return HealthyResult()


NVIDIA_FABRIC_MANAGER_HEALTH_CHECK: Final[LeafHealthCheck] = NvidiaFabricManagerHealthCheck()


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class DockerNvidiaCommunicationError(HealthCheckError):
    """Raised when there is a problem with communicating between Docker and NVIDIA."""

    suggested_remediation: str = "\n".join(("Try restarting the host or docker service to resolve this error.",))

    def create_fix_command(self, ip: str, user: str, port:int) -> str:
        return f"bash {HEALTH_CHECK_FIX_DIR}fix_docker_nvidia_communication.sh {ip} {user} {port}"


@attr.s(auto_attribs=True, frozen=True)
class DockerNvidiaCommunicationHealthCheck(LeafHealthCheck):
    def create_command(self) -> str:
        # see also: https://github.com/NVIDIA/nvidia-container-toolkit/issues/48
        return (
            f"docker run --rm --name infra_test_k_docker_nvidia_{str(datetime.datetime.utcnow().microsecond)} --gpus=all -i localhost:2899/ubuntu nvidia-smi || "
            + f"docker run --rm --name infra_test_docker_nvidia_{str(datetime.datetime.utcnow().microsecond)} --gpus=all -i ubuntu nvidia-smi 2>&1"
        )

    def validate_result(self, output: str, returncode: int) -> HealthCheckOutcome:
        if returncode != 0:
            return DockerNvidiaCommunicationError(
                message=f"The Docker NVIDIA communication health check gave a status of {output}"
                + f"Running `{self.create_command()}` on the host failed with return code {returncode}. \n"
                + "This occurs when docker and NVIDIA fail to communicate. \n"
            )
        return HealthyResult()


DOCKER_NVIDIA_COMMUNICATION_HEALTH_CHECK: Final[LeafHealthCheck] = DockerNvidiaCommunicationHealthCheck()

@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class DiskSpaceHealthCheckError(HealthCheckError):
    """Raised when there is not enough space remaining on the disks to be confident that they didn't contribute to some error"""

    suggested_remediation: str = "\n".join(
        (
            "Run tools like `df -h` and `du * -sch` to find out what's taking up all the disk space.",
            "Lack of space can cause lots of problems, including not being able to start docker containers.",
        )
    )


@attr.s(auto_attribs=True, frozen=True)
class DiskSpaceHealthCheck(LeafHealthCheck):
    def create_command(self) -> str:
        return "df --output=pcent,source,target"

    def validate_result(self, output: str, returncode: int) -> HealthCheckOutcome:
        if returncode != 0:
            return DiskSpaceHealthCheckError(
                message=f"Running `{self.create_command()}` on the host failed with return code {returncode}. \n"
                + "That really shouldn't happen, and probably indicates a severe failure in the filesystem.\n"
                + "Output:\n"
                + output
            )

        lines = output.splitlines()
        # skip the first line because it is a header
        for line in lines[1:]:
            pcent, fs_mount = line.split("%", maxsplit=2)
            pcent = pcent.strip()
            fs_mount = fs_mount.strip()
            if int(pcent) >= 95:
                return DiskSpaceHealthCheckError(
                    message=f"Running `{self.create_command()}` on the host displayed a filesystem mounted at {fs_mount} with less than 5% disk space remaining. \n"
                    + "Which quite likely means that disk pressure or a failure to write can or did contribute to whatever else went wrong on this host. \n"
                    + "Output:\n"
                    + output
                )

        return HealthyResult()


DISK_SPACE_HEALTH_CHECK: Final[LeafHealthCheck] = DiskSpaceHealthCheck()


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class AnsibleFactGatheringHangHealthCheckWarning(HealthCheckWarning):
    """Raised when there are multiple leftover sg_inq processes, which generally indicates that ansible fact gathering is hanging."""

    suggested_remediation: str = "\n".join(
        (
            "This is a known issue with ansible and can cause hosts to be unable to run ansible playbooks.",
        )
    )

    def create_fix_command(self, ip: str, user: str, port:int) -> str:
        return f"bash {HEALTH_CHECK_FIX_DIR}fix_ansible_fact_gathering_hang.sh {ip} {user} {port}"


@attr.s(auto_attribs=True, frozen=True)
class AnsibleFactGatheringHangHealthCheck(LeafHealthCheck):
    def create_command(self) -> str:
        # If we find out that this is flaky we could also parse out the start times of the processes and only include ones sufficiently old (5m?).
        # But that's much more complicated and this should work for now.
        return "ps aux | grep -v grep | grep sg_inq | wc -l"

    def validate_result(self, output: str, returncode: int) -> HealthCheckOutcome:
        if returncode != 0:
            return AnsibleFactGatheringHangHealthCheckWarning(
                message=f"The command failed with return code {returncode}."
            )
        if int(output) > 1:
            return AnsibleFactGatheringHangHealthCheckWarning(
                message=f"Found {output} leftover sg_inq processes, which usually indicates the host hasn't been properly ansible-ed"
            )
        return HealthyResult()


ANSIBLE_FACT_GATHERING_HANG_HEALTH_CHECK: Final[LeafHealthCheck] = AnsibleFactGatheringHangHealthCheck()


# A list of ranges of XIDs that are known to be hardware errors.
# https://docs.nvidia.com/deploy/xid-errors/index.html
_HARDWARE_XID_ERROR_RANGES: Final[Container[int]] = (
    48,
    *range(56, 59),
    *range(62, 65),
    *range(68, 78),
    *range(79, 87),
    *range(88, 90),
    92,
    *range(94, 106),
    *range(110, 121),
    *range(122, 126),
)


_DMESG_XID_REGEX: Optional[re.Pattern] = None
_DMESG_SXID_REGEX: Optional[re.Pattern] = None
_DMESG_PCIE_REGEX: Optional[re.Pattern] = None
_PARSE_GPU_NUM_REGEX: Optional[re.Pattern] = None


def _parse_xid_from_dmesg_line(line: str) -> Optional[int]:
    """Parses the XID from a line of output from the `dmesg` command.

    Examples:
    >>> line = "[2397166.983093] NVRM: Xid (PCI:0000:0f:00): 94, pid=2503421, name=python3, Ch 00000008"
    >>> _parse_xid_from_dmesg_line(line)
    94
    >>> line = "[2397166.923949] NVRM: Xid (PCI:0000:0f:00): 64, pid='<unknown>', name=<unknown>, Row Remapper Error: (0x00000002eebc4100) - All reserved rows for bank are remapped"
    >>> _parse_xid_from_dmesg_line(line)
    64
    >>> line = "[2397167.453421] Unrelated line: 123"
    >>> _parse_xid_from_dmesg_line(line) is None
    True
    """
    pattern = r"NVRM: Xid \(PCI:[\w:]+?\): (\d+)"
    global _DMESG_XID_REGEX
    if _DMESG_XID_REGEX is None:
        _DMESG_XID_REGEX = re.compile(pattern)
    match = re.search(_DMESG_XID_REGEX, line)
    if match is None:
        return None
    return int(match.group(1))


@attr.s(auto_attribs=True, frozen=True)
class _DmesgXIDHealthCheck(LeafHealthCheck):
    def create_command(self) -> str:
        # We could also filter messages using grep but dealing with status codes correctly makes the command more complicated.
        return "sudo dmesg --level warn"

    def validate_result(self, output: str, returncode: int) -> HealthCheckOutcome:
        if returncode != 0:
            return GpuHealthCheckError(message=f"The `dmesg` command failed with return code {returncode}.")
        xids = (_parse_xid_from_dmesg_line(line) for line in output.splitlines())
        hardware_error_xids = [xid for xid in xids if xid is not None and xid in _HARDWARE_XID_ERROR_RANGES]
        if hardware_error_xids:
            return GpuHealthCheckError(message=f"Found XIDs in the hardware error ranges: {hardware_error_xids}")
        return HealthyResult()


DMESG_XID_HEALTH_CHECK: Final[LeafHealthCheck] = _DmesgXIDHealthCheck()

def _parse_limited_hardware_addr_from_dmesg_line(line: str) -> Optional[Tuple[str, str]]:
    """Parses the hardware addr with limited pcie from a line of output from the `dmesg` command.

    Examples:
    >>> line = "[   18.617069] mlx5_core 0000:9c:00.0: 252.056 Gb/s available PCIe bandwidth, limited by 32.0 GT/s PCIe x8 link at 0000:97:01.0 (capable of 504.112 Gb/s with 32.0 GT/s PCIe x16 link)"
    >>> _parse_limited_hardware_addr_from_dmesg_line(line)
    ('0000:9c:00.0', '252.056 Gb/s available PCIe bandwidth, limited by 32.0 GT/s PCIe x8 link at 0000:97:01.0 (capable of 504.112 Gb/s with 32.0 GT/s PCIe x16 link)')
    >>> line = "[    8.328180] pci 0000:02:00.0: 4.000 Gb/s available PCIe bandwidth, limited by 5.0 GT/s PCIe x1 link at 0000:00:0c.0 (capable of 8.000 Gb/s with 5.0 GT/s PCIe x2 link)    >>> _parse_limited_hardware_addr_from_dmesg_line(line)"
    >>> _parse_limited_hardware_addr_from_dmesg_line(line) is None
    True
    """
    pattern = r"mlx5_core (\w{4}:\w{2}:\w{2}.\w): (.*available PCIe bandwidth, limited.*)"
    global _DMESG_PCIE_REGEX
    if _DMESG_PCIE_REGEX is None:
        _DMESG_PCIE_REGEX = re.compile(pattern)
    match = re.search(_DMESG_PCIE_REGEX, line)
    if match is None:
        return None
    return match.group(1), match.group(2)


def _parse_sxid_error_code_from_dmesg_line(line: str) -> Optional[Tuple[str, str]]:
    """Parses the hardward addr with limited pcie from a line of output from the `dmesg` command.

    Examples:
    >>> line = "[186965.348898] nvidia-nvswitch2: SXid (PCI:0000:85:00.0): 12028, Non-fatal, Link 61 egress non-posted PRIV error (First)"
    >>> _parse_sxid_error_code_from_dmesg_line(line)
    ('0000:85:00.0', '12028')
    >>> line = "[    8.328180] pci 0000:02:00.0: 4.000 Gb/s available PCIe bandwidth, limited by 5.0 GT/s PCIe x1 link at 0000:00:0c.0 (capable of 8.000 Gb/s with 5.0 GT/s PCIe x2 link)"
    >>> _parse_sxid_error_code_from_dmesg_line(line) is None
    True
    """

    pattern = r"nvidia-nvswitch.*: SXid \(PCI:(\w*:\w*:\w*\.\w*)\): (\d+)"
    global _DMESG_SXID_REGEX
    if _DMESG_SXID_REGEX is None:
        _DMESG_SXID_REGEX = re.compile(pattern)
    match = re.search(_DMESG_SXID_REGEX, line)
    if match is None:
        return None
    return match.group(1), match.group(2)


def _match_hardware_addr_with_mlx(output: str, bad_hardware_addrs: Iterable[Tuple[str, str]]) -> Tuple[str, ...]:
    """Parses the lines that contain information connection hardware addr and gpu number

    Examples:
    >>> output = "GPU#7 mlx_11 0000:dc:00.0"
    >>> bad_hardware_addrs = [("0000:dc:00.0", "[message]")]
    >>> _match_hardware_addr_with_mlx(output, bad_hardware_addrs)
    ('GPU#7 mlx_11 0000:dc:00.0: [message]',)
    >>> output = "ETH#10 mlx_13 0000:dc:00.0"
    >>> bad_hardware_addrs = [("0000:dc:00.0", "[message]")]
    >>> len(_match_hardware_addr_with_mlx(output, bad_hardware_addrs)) == 0
    True
    """
    hardware_addr_to_mlx_info: Dict[str, Tuple[bool, str]] = dict()

    pattern = r"(.*#\d+ mlx_\d+) (\w{4}:\w{2}:\w{2}\.\w)"
    global _PARSE_GPU_NUM_REGEX
    if _PARSE_GPU_NUM_REGEX is None:
        _PARSE_GPU_NUM_REGEX = re.compile(pattern)
    for line in output.splitlines():
        match = re.search(_PARSE_GPU_NUM_REGEX, line)
        if match is not None:
            device_info, hardware_addr = match.group(1), match.group(2)
            is_gpu = "GPU" in device_info
            hardware_addr_to_mlx_info[hardware_addr] = (is_gpu, device_info.strip())

    bad_mlx_info = []
    for hardware_addr, message in bad_hardware_addrs:
        if hardware_addr in hardware_addr_to_mlx_info:
            is_gpu, gpu_info = hardware_addr_to_mlx_info[hardware_addr]
            if is_gpu:
                bad_mlx_info.append(f"{gpu_info} {hardware_addr}: {message}")
        else:
            bad_mlx_info.append(f"Unknown device with hardware address {hardware_addr}")
    return tuple(bad_mlx_info)


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class PcieLimitedHealthCheckError(HealthCheckError):
    """Raised when dmesg indicates that PCIe is limiting bandwidth on this machine"""

    suggested_remediation: str = "\n".join(
        (
            "Remove this machine from multinode training.",
            "Try rebooting the machine.",
            "If that doesn't work a support ticket may be needed.",
        )
    )

    def create_fix_command(self, ip: str, user: str, port:int) -> str:
        return f"bash {HEALTH_CHECK_FIX_DIR}fix_pcie_limited.sh {ip} {user} {port}"


# This is hard-coded to Imbue's mlx index naming, you'll probably need to change it
_DMESG_LIMITED_PCIE_COMMAND: Final[
    str
] = """\
set -euo pipefail

# Find the Mellanox log entries for Mellanox devices with limited PCIe bandwidth.
dmesg | ( grep mlx5_core || [[ $? == 1 ]] ) | ( grep -i limited || [[ $? == 1 ]] )

# Print the hardware address for each infiniband and ethernet device.
gpu_idx=0
for ib_idx in 0 3 4 5 6 9 10 11 1 2 7 8
do
    # We only (currently) care about the infiniband devices
    # The indices are a little weird:
    case $ib_idx in 0|3|4|5|6|9|10|11)
        echo -en "GPU#${gpu_idx} mlx_${ib_idx} "
        ;;
    *)
        echo -en "ETH#${gpu_idx} mlx_${ib_idx} "
        ;;
    esac
    gpu_idx=$(($gpu_idx + 1))
    readlink -f "/sys/class/infiniband/mlx5_${ib_idx}/device" | awk -F'/' '{ print $NF }'
done
"""


@attr.s(auto_attribs=True, frozen=True)
class _DmesgLimitedPcieHealthCheck(LeafHealthCheck):
    """A health check that looks for PCIe bandwidth limitations in the dmesg output.
    This health check is specific to Mellanox devices as limited Ethernet bandwidth doesn't affect our training.
    """
    def create_command(self) -> str:
        return _DMESG_LIMITED_PCIE_COMMAND

    def validate_result(self, output: str, returncode: int) -> HealthCheckOutcome:
        if returncode != 0:
            return PcieLimitedHealthCheckError(message=f"The `dmesg` command failed with return code {returncode}.")
        limited_hardware_addresses = remove_none(
            _parse_limited_hardware_addr_from_dmesg_line(line) for line in output.splitlines()
        )
        limited_mlx_info = _match_hardware_addr_with_mlx(output, limited_hardware_addresses)
        if len(limited_mlx_info) > 0:
            limited_mlx = ",".join(limited_mlx_info)
            return PcieLimitedHealthCheckError(message=f"Found PCIe is limited for mlx: {limited_mlx}")
        return HealthyResult()


DMESG_PCIE_HEALTH_CHECK: Final[LeafHealthCheck] = _DmesgLimitedPcieHealthCheck()


class OuterTimeoutFailure(HealthCheckIncomplete):
    """
    Filled in for parallel healthcheck runs that have an outer timeout that is hit before the inner timeout.
    Generally shouldn't exist, but we still need to handle the case because of nondeterminism.
    """

    def display(self, indentation: int = 0) -> str:
        """Produce a string with a human-readable summary of the central service error that has occurred.

        Accepts an optional number of spaces by which to indent each line of the message.
        """
        message = type(self).__name__ + ":"
        if self.message is not None:
            message += "\nError: " + self.message

        prefix = indentation * " "
        message = "\n".join(f"{prefix}{line}" for line in message.splitlines())
        return message



@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class InfinibandHealthCheckError(HealthCheckError):
    """
    Raised when there are errors with the NICs or IB connectivity on this machine.
    """

    suggested_remediation: str = "\n".join(
        ("Remove the host for multi-node training.", "Flag as it may need a support ticket from Dell.")
    )


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class InfinibandHealthCheckWarning(HealthCheckSilencedWarning):
    """
    Raised when there is increased Infiniband error rate on one of the NICs on this machine.
    """

    suggested_remediation: str = "\n".join(
        (
            "Review error counters and remove from multi-node training if necessary",
            "Flag as it may need a support ticket.",
        )
    )


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class InfinibandDriverHealthCheckError(HealthCheckError):
    """Raised when infiniband drivers are out out of date for this machine"""

    suggested_remediation: str = "\n".join(
        (
            "Update relevant drivers",
        )
    )


@attr.s(auto_attribs=True, frozen=True)
class _IbErrorHealthCheck(LeafHealthCheck):
    """
    Checks IB/NIC diagnostics and validates that all of them are present with no errors.
    But first checks driver versions and throws errors immediately if any of the drivers are outdated.
    `hca_self_test` can be downloaded from here: https://docs.nvidia.com/networking/display/mlnxofedv461000/installing+mellanox+ofed
    """

    def create_command(self) -> str:
        command_parts = (
            "sudo flock -w 90 -F /usr/bin/hca_self_test.ofed /usr/bin/hca_self_test.ofed  </dev/null 2>&1",
            # Remove color codes
            r"sed 's/\x1B\[[0-9;]\{1,\}[A-Za-z]//g'",
            # Replace valid UUIDs with a predictable string.
            # A valid UUID looks like: 94:6d:ae:03:00:af:de:7c
            r"sed 's/[0-9a-f]\{2\}\(:[0-9a-f]\{2\}\)\{7\}/VALID_UUID/g'",
            # The script produces output for both Infiniband HCA cards, as well as NICs.
            # We don't care about the output for NICs.
            "grep -v 'NIC'",
            # We ignore Ethernet lines that are down
            "grep -v '(Ethernet)'",
            "grep -v 'tput: unknown terminal'",
        )
        command = " | ".join(command_parts)
        return command

    def _find_nonmatching_lines(
        self, output: str, expected_dict: Dict[str, str], expected_total_count: int
    ) -> List[str]:
        """
        Parses the hca_self_test.ofed output style and searches it for key-value pairs.
        If the label of any line in the output matches a key of expected_dict but the line value does not match the corresponding dict value,
        a corresponding error is returned in the resulting array.
        """

        total_passes = 0
        errors = []
        for line in output.splitlines():
            split_line = re.split(r"\.{2,}", line)
            if len(split_line) != 2:
                continue
            label, value = [p.strip() for p in split_line]
            for expected_line_label, expected_line_value in expected_dict.items():
                if expected_line_label in label:
                    if expected_line_value != value and expected_line_value != value.split(" ", 1)[0]:
                        errors.append(f"{label}:{value}")
                    else:
                        total_passes += 1
                    break
        if len(errors) == 0 and total_passes != expected_total_count:
            errors.append(
                f"Unexpected number of lines {total_passes}, not {expected_total_count}, gpu may be missing. Samples:"
            )
            errors += output.splitlines()[:3]
        return errors

    def _validate_driver_versions(self, output: str) -> HealthCheckOutcome:
        lines = [l for l in output.splitlines() if "Host Driver Version" in l or "Firmware" in l]

        expected_line_label_to_value = {
            "Host Driver Version": EXPECTED_CONFIG["infiniband_error"]["driver_versions"]["Host Driver Version"],
            "Host Driver Version": EXPECTED_CONFIG["infiniband_error"]["driver_versions"]["Host Driver Version"],
            "Firmware on CA": EXPECTED_CONFIG["infiniband_error"]["driver_versions"]["Firmware on CA"],  # 8x
        }
        expected_total_passes = EXPECTED_CONFIG["infiniband_error"]["driver_versions"]["total_passes"]

        errors = self._find_nonmatching_lines(output, expected_line_label_to_value, expected_total_passes)
        if len(errors) > 0:
            error_message = "\n".join([f"{len(errors)} driver mismatch(es) detected"] + errors)
            return InfinibandDriverHealthCheckError(message=error_message)
        return HealthyResult()

    def _validate_hcas_are_active(self, output: str) -> HealthCheckOutcome:
        expected_line_label_to_value = {
            "Port State": EXPECTED_CONFIG["infiniband_error"]["hcas"]["Port State"],  # 8x
            "Node GUID": EXPECTED_CONFIG["infiniband_error"]["hcas"]["Node GUID"],  # 8x
            "Host Driver Initialization": "PASS",  # 1x
            "Host Driver RPM": "PASS",  # 1x
            # "Kernel Syslog": "PASS",  # 1x
            "PCI Device": "PASS",  # 1x
        }
        expected_total_passes = EXPECTED_CONFIG["infiniband_error"]["hcas"]["total_passes"]

        errors = self._find_nonmatching_lines(output, expected_line_label_to_value, expected_total_passes)
        if len(errors) > 0:
            error_message = "\n".join([f"{len(errors)} error(s) detected"] + errors)
            return InfinibandHealthCheckError(message=error_message)
        return HealthyResult()

    def _validate_error_counters(self, output: str) -> HealthCheckOutcome:
        expected_line_label_to_value = {
            "Error Counter Check": "PASS",
        }
        expected_total_passes = EXPECTED_CONFIG["infiniband_error"]["error_counters"]["total_passes"]

        errors = self._find_nonmatching_lines(output, expected_line_label_to_value, expected_total_passes)
        if len(errors) > 0:
            error_message = "\n".join([f"{len(errors)} error counter(s) too high"] + errors)
            return InfinibandHealthCheckWarning(message=error_message)
        return HealthyResult()

    def validate_result(self, output: str, returncode: int) -> HealthCheckOutcome:
        if returncode != 0:
            return InfinibandHealthCheckError(
                message=f"`hca_self_test` command failed with returncode {returncode} and output {output}"
            )

        driver_check_outcome = self._validate_driver_versions(output)
        error_check_outcome = self._validate_hcas_are_active(output)
        error_counters_outcome = self._validate_error_counters(output)

        return make_compound_error([driver_check_outcome, error_check_outcome, error_counters_outcome])


IB_HEALTH_CHECK: Final[LeafHealthCheck] = _IbErrorHealthCheck()

MIN_NVLINK_ERRORS_TO_WARN: Final[int] = 5


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class NvLinkHealthCheckWarning(HealthCheckSilencedWarning):
    """Raised when there are nvlink errors for this machine.
    These errors don't seem to impact training so this is simply a warning but not a failure."""

    suggested_remediation: str = "\n".join(
        (
            "Check dmesg for any weird issues",
            "Keep monitoring this host for issues and potential slowdowns",
        )
    )


@attr.s(auto_attribs=True, frozen=True)
class _NvlinkHealthCheck(LeafHealthCheck):
    """
    Validates that nvlink errors are not present on each of the 8 gpus.
    """

    def create_command(self) -> str:
        command_parts = ("nvidia-smi nvlink -e", 'grep -v -e "Errors: 0" -e "^ *$"')
        command = " | ".join(command_parts)
        return command

    def validate_result(self, output: str, returncode: int) -> HealthCheckOutcome:
        if returncode != 0:
            return NvLinkHealthCheckWarning(
                message=f"`nvidia-smi nvlink` command failed with returncode {returncode} and output {output}"
            )
        errors = []
        current_gpu = None
        total_errors = 0
        for line in output.splitlines():
            line = line.strip()
            if len(line) == 0:
                continue
            if line.startswith("GPU"):
                current_gpu = line.split(":", 1)[0]
            else:
                num_errors_str = line.rsplit(":", 1)[1].strip()
                try:
                    num_errors = int(num_errors_str)
                    total_errors += num_errors
                except ValueError:
                    total_errors += MIN_NVLINK_ERRORS_TO_WARN
                errors.append(f"{current_gpu}, {line.strip()}")
        if len(errors) > 0 and total_errors >= MIN_NVLINK_ERRORS_TO_WARN:
            error_message = f"{len(errors)} error(s) detected"
            return NvLinkHealthCheckWarning(message=error_message)
        return HealthyResult()


NVLINK_HEALTH_CHECK: Final[LeafHealthCheck] = _NvlinkHealthCheck()


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class NvLinkHealthCheckError(HealthCheckError):
    """Raised when there are nvlink errors in dmesg."""

    suggested_remediation: str = "\n".join(("Empirically this is probably not an issue, but monitor for slowness",))


@attr.s(auto_attribs=True, frozen=True)
class _DmesgNvLinkHealthCheck(LeafHealthCheck):
    def create_command(self) -> str:
        return "sudo cat /var/log/dmesg && sudo dmesg"

    def validate_result(self, output: str, returncode: int) -> HealthCheckOutcome:
        if returncode != 0:
            return DmesgHealthCheckWarning(message=f"The `dmesg` command failed with return code {returncode}.")
        lines = output.splitlines()
        nvswitch_strs = ("nvswitch", "nvlink")
        failure_strs = ("error", "fatal", "failed", "assert", "timed out", "timeout")
        for line in lines:
            sxid_error = _parse_sxid_error_code_from_dmesg_line(line)
            if sxid_error is not None:
                switch_pci_bdf, sxid_value = sxid_error
                if sxid_value not in WHITELISTED_NVSWITCH_SXID_ERRORS:
                    return NvLinkHealthCheckError(message=f"Found nvswitch sxid error: {line}")
            elif any(option in line.lower() for option in nvswitch_strs) and any(
                option in line.lower() for option in failure_strs
            ):
                return NvLinkHealthCheckError(message=f"Found nvswitch error: {line}")
        return HealthyResult()


DMESG_NVLINK_HEALTH_CHECK: Final[LeafHealthCheck] = _DmesgNvLinkHealthCheck()


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class NvLinkStatusHealthCheckError(HealthCheckError):
    """Raised when there are nvlink connections that are disabled."""

    suggested_remediation: str = "\n".join(("File a support ticket.",))


@attr.s(auto_attribs=True, frozen=True)
class _NvlinkStatusHealthCheck(LeafHealthCheck):
    """
    Validates that nvlink connections are active on each of the 8 gpus.
    """

    def create_command(self) -> str:
        command_parts = ("nvidia-smi nvlink -s", 'grep "<inactive>"')
        command = " | ".join(command_parts)
        return command

    def validate_result(self, output: str, returncode: int) -> HealthCheckOutcome:
        if returncode == 1:
            return HealthyResult()
        if returncode != 0:
            return NvLinkStatusHealthCheckError(
                message=f"`nvidia-smi nvlink` command failed with returncode {returncode} and output {output}"
            )
        if len(output.strip()) > 0:
            return NvLinkStatusHealthCheckError(
                message=f"Inactive NVLink connections detected: {output.strip()[:100]}"
            )
        return HealthyResult()


NVLINK_STATUS_HEALTH_CHECK: Final[LeafHealthCheck] = _NvlinkStatusHealthCheck()


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class GdrEnabledHealthCheckWarning(HealthCheckWarning):
    """
    Raised when GDR is not enabled for this machine.
    """

    suggested_remediation: str = "\n".join(
        ("GDR is not enabled for this machine. Run `sudo modprobe nvidia-peermem`",
         "Refer here: https://download.nvidia.com/XFree86/Linux-x86_64/535.183.01/README/nvidia-peermem.html",)
    )


@attr.s(auto_attribs=True, frozen=True)
class _GdrEnabledHealthCheck(LeafHealthCheck):
    """
    Validates that GDR is enabled on the machine.
    """

    def create_command(self) -> str:
        return "sudo lsmod | grep peermem"

    def validate_result(self, output: str, returncode: int) -> HealthCheckOutcome:
        if returncode == 1:
            return GdrEnabledHealthCheckWarning(message=f"`GPU direct rdma is not enabled on this machine.")
        if returncode != 0:
            return GdrEnabledHealthCheckWarning(
                message=f"`sudo lsmod | grep peermem` command failed with returncode {returncode} and output {output}"
            )

        if output.strip() == "":
            return GdrEnabledHealthCheckWarning(message=f"`GPU direct rdma is not enabled on this machine.")
        return HealthyResult()


GDR_ENABLED_HEALTH_CHECK: Final[LeafHealthCheck] = _GdrEnabledHealthCheck()


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class ZPoolHealthCheckWarning(HealthCheckWarning):
    """
    Raised when zpool is not configured properly.
    """

    suggested_remediation: str = "\n".join(
        ("zpool is not configured correctly for this machine.",)
    )

    def create_fix_command(self, ip: str, user: str, port:int) -> str:
        return f"bash {HEALTH_CHECK_FIX_DIR}fix_zpool.sh {ip} {user} {port}"


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class ZPoolHealthCheckError(HealthCheckError):
    """
    Raised when zpool command is not found for the machine.
    """

    suggested_remediation: str = "\n".join(
        (
            "`zpool` command not found, or returning degraded/invalid checksums.",
        )
    )

    def create_fix_command(self, ip: str, user: str, port:int) -> str:
        return f"bash {HEALTH_CHECK_FIX_DIR}fix_zpool.sh {ip} {user} {port}"


@attr.s(auto_attribs=True, frozen=True)
class _ZPoolHealthCheck(LeafHealthCheck):
    """
    Validates that zpool is configured for this machine.
    """

    def create_command(self) -> str:
        return "sudo zpool status zpool | sed '1,/config:/d'"

    def validate_result(self, output: str, returncode: int) -> HealthCheckOutcome:
        if returncode == 1:
            return ZPoolHealthCheckError(message=f"`sudo zpool status zpool` command not found, output {output}")
        if returncode != 0:
            return ZPoolHealthCheckWarning(
                message=f"`sudo zpool status zpool` command failed with returncode {returncode} and output {output}"
            )

        set_uuids = set()
        for line in output.splitlines():
            if not line.startswith("\t"):
                continue
            entries = line.strip().split()
            if len(entries) < 5 or entries == ["NAME", "STATE", "READ", "WRITE", "CKSUM"]:
                continue
            name, state, read, write, cksum = entries
            if name.startswith("nvme-eui."):
                set_uuids.add(line.strip())
            elif name.startswith("nvme-Dell"):
                set_uuids.add(line.strip())

            if state != "ONLINE":
                return ZPoolHealthCheckError(
                    message=f"{name} has invalid state {state}. Detailed status of {','.join(entries)}"
                )
            if read != "0" or write != "0" or cksum != "0":
                return ZPoolHealthCheckError(
                    message=f"{name} has invalid values. Detailed status of {','.join(entries)}"
                )

        if len(set_uuids) < 6:
            return ZPoolHealthCheckWarning(message=f"zpool uuids are not configured correctly for this machine.")
        return HealthyResult()


ZPOOL_HEALTH_CHECK: Final[LeafHealthCheck] = _ZPoolHealthCheck()


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class VBiosHealthCheckWarning(HealthCheckWarning):
    """
    Raised when VBIOS versions are not consistent.
    """

    suggested_remediation: str = "\n".join(("VBIOS versions are not consistent. Update to the appropriate version.",))

    def create_fix_command(self, ip: str, user: str, port:int) -> str:
        return f"bash {HEALTH_CHECK_FIX_DIR}fix_vbios_version.sh {ip} {user} {port}"


@attr.s(auto_attribs=True, frozen=True)
class _VBiosVersionHealthCheck(LeafHealthCheck):
    """
    Validates that VBIOS is on the correct version for this machine.
    """

    def create_command(self) -> str:
        return "nvidia-smi -q | grep Version | sort | uniq -c | sort"

    def validate_result(self, output: str, returncode: int) -> HealthCheckOutcome:
        if returncode != 0:
            return VBiosHealthCheckWarning(
                message=f"`nvidia-smi` command failed with returncode {returncode} and output {output}"
            )
        expected_output = [
            ["1 CUDA Version", EXPECTED_CONFIG["vbios"]["CUDA Version"]],
            ["1 Driver Version", EXPECTED_CONFIG["vbios"]["Driver Version"]],
            ["8 Image Version", EXPECTED_CONFIG["vbios"]["Image Version"]],
            ["8 GSP Firmware Version", EXPECTED_CONFIG["vbios"]["GSP Firmware Version"]],
            ["8 Inforom Version"],
            ["8 VBIOS Version", EXPECTED_CONFIG["vbios"]["VBIOS Version"]],
        ]
        output_parsed = [[re.sub('\s{2,}', ' ', field.strip()) for field in line.split(":")] for line in output.splitlines()]
        silenced_items = []
        expected_output = sorted([item for item in expected_output if item[0] not in silenced_items])
        output_parsed = sorted([item for item in output_parsed if item[0] not in silenced_items])
        if output_parsed != expected_output:
            for i in range(len(output_parsed)):
                if output_parsed[i] != expected_output[i]:
                    return VBiosHealthCheckWarning(
                        message=f"Nvidia versions are not consistent at {i}. Expected {expected_output[i]}, got {output_parsed[i]}"
                    )
            return VBiosHealthCheckWarning(message=f"VBIOS versions are not consistent.")
        return HealthyResult()


VBIOS_VERSION_HEALTH_CHECK: Final[LeafHealthCheck] = _VBiosVersionHealthCheck()


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class SwapHealthCheckWarning(HealthCheckSilencedWarning):
    """
    Raised when swap is on for the machine.
    """

    suggested_remediation: str = "\n".join(("Swap is on, turn it off.",))

    def create_fix_command(self, ip: str, user: str, port:int) -> str:
        return f"bash {HEALTH_CHECK_FIX_DIR}fix_swap.sh {ip} {user} {port}"


@attr.s(auto_attribs=True, frozen=True)
class _SwapHealthCheck(LeafHealthCheck):
    """
    Validates that swap is off.
    """

    def create_command(self) -> str:
        return "swapon --show"

    def validate_result(self, output: str, returncode: int) -> HealthCheckOutcome:
        if returncode != 0:
            return SwapHealthCheckWarning(
                message=f"`swapon --show` command failed with returncode {returncode} and output {output}"
            )
        if output != "":
            return SwapHealthCheckWarning(message=f"Swap is on.")
        return HealthyResult()


SWAP_HEALTH_CHECK: Final[LeafHealthCheck] = _SwapHealthCheck()


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class DockerInfoHealthCheckError(HealthCheckError):
    """
    Raised when docker info has a critical issue.
    """

    suggested_remediation: str = "\n".join(("Try restarting the host or docker service to resolve this error.",))

    def create_fix_command(self, ip: str, user: str, port:int) -> str:
        return f"bash {HEALTH_CHECK_FIX_DIR}fix_docker_info.sh {ip} {user} {port}"


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class DockerInfoHealthCheckWarning(HealthCheckWarning):
    """
    Raised when docker info has a discrepancy.
    """

    suggested_remediation: str = "\n".join(("Try restarting the host or docker service to resolve this error.",))

    def create_fix_command(self, ip: str, user: str, port:int) -> str:
        return f"bash {HEALTH_CHECK_FIX_DIR}fix_docker_info.sh {ip} {user} {port}"


@attr.s(auto_attribs=True, frozen=True)
class _DockerInfoHealthCheck(LeafHealthCheck):
    """
    Validates that docker info is consistent.

    Raises an error if the storage driver is not ZFS.
    Raises a warning if the versions are not updated.
    """

    def create_command(self) -> str:
        return "docker info --format json"

    def _confirm_field_info(
        self, docker_info: List | Dict, expected_info: List | Dict, docker_exist: bool = True
    ) -> Dict:
        incorrect_fields = {}
        if isinstance(expected_info, list):
            if not isinstance(docker_info, list):
                incorrect_fields = self._confirm_field_info(expected_info[0], expected_info[0], False)
            elif len(docker_info) != 1 or len(expected_info) != 1:
                incorrect_fields = self._confirm_field_info(expected_info[0], expected_info[0], False)
            else:
                incorrect_fields = self._confirm_field_info(docker_info[0], expected_info[0], docker_exist)
            return incorrect_fields
        for field in expected_info:
            if field not in docker_info:
                incorrect_fields[field] = (None, expected_info[field])
                continue
            if not isinstance(expected_info[field], (list, dict)):
                if not docker_exist:
                    incorrect_fields[field] = (None, expected_info[field])
                elif docker_info[field] != expected_info[field]:
                    incorrect_fields[field] = (docker_info[field], expected_info[field])
                continue
            incorrect_subfields = self._confirm_field_info(docker_info[field], expected_info[field], docker_exist)
            for subfield in incorrect_subfields:
                incorrect_fields[f"{field}/{subfield}"] = incorrect_subfields[subfield]
        return incorrect_fields

    def validate_result(self, output: str, returncode: int) -> HealthCheckOutcome:
        if returncode != 0:
            return DockerInfoHealthCheckError(
                message=f"`docker info` command failed with returncode {returncode} and output {output}"
            )

        try:
            docker_info = json.loads(output)
        except ValueError as e:
            return DockerInfoHealthCheckError(message=repr(e))

        expected_error_info = {"Driver": EXPECTED_CONFIG["docker"]["expected_error_info"]["Driver"]}

        expected_warning_info = EXPECTED_CONFIG["docker"]["expected_warning_info"]

        mismatched_error_info = self._confirm_field_info(docker_info, expected_error_info)
        if len(mismatched_error_info) > 0:
            return DockerInfoHealthCheckWarning(message=f"Docker Storage Driver is not ZFS")

        mismatched_warning_info = self._confirm_field_info(docker_info, expected_warning_info)
        if len(mismatched_warning_info) > 0:
            return DockerInfoHealthCheckWarning(message=f"Docker version does not match the expected version")

        return HealthyResult()


DOCKER_INFO_HEALTH_CHECK: Final[LeafHealthCheck] = _DockerInfoHealthCheck()




@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class FlintVersionHealthCheckWarning(HealthCheckError):
    """
    Raised when flint finds incorrect versions for card or transceiver.
    """

    suggested_remediation: str = "\n".join(("Flint found inconsistent versions for card or transceiver.",
                                            "Try upgrading the firmware on the card or transceiver."))

    def create_fix_command(self, ip: str, user: str, port:int) -> str:
        return f"bash {HEALTH_CHECK_FIX_DIR}fix_flint_version.sh {ip} {user} {port}"


@attr.s(auto_attribs=True, frozen=True)
class _FlintVersionHealthCheck(LeafHealthCheck):
    """
    Validates that flint finds the correct versions for card and transceiver.
    """

    def create_command(self) -> str:
        command_parts = (
            f"for CARD in {' '.join(get_hca_cards())}" ,
            "do",
            "    echo CARD: $CARD",
            "    sudo flint -d $CARD query",
            "    sudo flint -d $CARD --linkx --downstream_device_ids 1 query",
            "done",
        )
        command = "\n".join(command_parts)
        return command

    def validate_result(self, output: str, returncode: int) -> HealthCheckOutcome:
        if returncode != 0:
            return FlintVersionHealthCheckWarning(message=f"The `flint` command failed with returncode {returncode}")

        expected_fields = EXPECTED_CONFIG["flint"]["expected_fields"]

        expected_FW_possibilities = EXPECTED_CONFIG["flint"]["expected_FW_possibilities"]
        num_FW_possibilities = len(next(iter(expected_FW_possibilities.values())))
        for possible_values in expected_FW_possibilities.values():
            assert len(possible_values) == num_FW_possibilities

        output_lines_cards = output.split("CARD: ")
        for card_lines in output_lines_cards:
            if card_lines == "":
                continue
            lines = card_lines.splitlines()
            card = lines[0]
            output_FW_values: Dict[str, Optional[str]] = dict()
            for field in expected_FW_possibilities:
                output_FW_values[field] = None
            for line_idx in range(len(lines)):
                line_parsed = lines[line_idx].split(":")
                if len(line_parsed) < 2:
                    continue
                tag = line_parsed[0].strip()
                if tag in expected_fields:
                    expected_values = expected_fields[tag]
                    output_values = [line_parsed[1]] + lines[line_idx + 1 : line_idx + len(expected_values)]
                    output_values = [value.strip() for value in output_values]
                    if output_values != expected_values:
                        expected_string = "\n".join(expected_values)
                        output_string = "\n".join(output_values)
                        return FlintVersionHealthCheckWarning(
                            message=f"Flint found inconsistent versions for {tag} for card {card}. Expected {expected_string}, got {output_string}"
                        )
                if tag in expected_FW_possibilities:
                    output_FW_values[tag] = line_parsed[1].strip()

            FW_possibility_satisfied = False
            for i in range(num_FW_possibilities):
                if all(
                    output_FW_values[field] == possibilities[i]
                    for field, possibilities in expected_FW_possibilities.items()
                ):
                    FW_possibility_satisfied = True
                    break
            if not FW_possibility_satisfied:
                return FlintVersionHealthCheckWarning(
                    message=f"Flint found inconsistent versions for FW Version on card {card}."
                )
        return HealthyResult()


FLINT_VERSION_HEALTH_CHECK: Final[LeafHealthCheck] = _FlintVersionHealthCheck()



@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class UbuntuVersionHealthCheckWarning(HealthCheckSilencedWarning):
    """
    Raised when Ubuntu is not on the correct version.
    """

    suggested_remediation: str = "\n".join(
        ("Ubuntu is not on the correct version. Update to the appropriate version.",)
    )


@attr.s(auto_attribs=True, frozen=True)
class _UbuntuVersionHealthCheck(LeafHealthCheck):
    """
    Validates that Ubuntu is on the correct version for this machine.
    """

    def create_command(self) -> str:
        return "lsb_release -a"

    def validate_result(self, output: str, returncode: int) -> HealthCheckOutcome:
        if returncode != 0:
            return UbuntuVersionHealthCheckWarning(
                message=f"`lsb_release -a` command failed with returncode {returncode} and output {output}"
            )
        for line in output.splitlines():
            fields = line.split(":")
            if len(fields) < 2:
                continue
            tag = fields[0].lower().strip()
            state = fields[1].lower().strip()
            if tag == "distributor id" and state != EXPECTED_CONFIG["ubuntu"]["distributor id"]:
                return UbuntuVersionHealthCheckWarning(message=f"Distributor ID is {state}, not {EXPECTED_CONFIG['ubuntu']['distributor id']}")
            if tag == "description" and state != EXPECTED_CONFIG["ubuntu"]["description"]:
                return UbuntuVersionHealthCheckWarning(
                    message=f"Distributor description is {state}, not ubuntu 22.04.3 lts"
                )
            if tag == "release" and state != EXPECTED_CONFIG["ubuntu"]["release"]:
                return UbuntuVersionHealthCheckWarning(message=f"Ubuntu release version is {state}, not {EXPECTED_CONFIG['ubuntu']['release']}")
            if tag == "codename" and state != EXPECTED_CONFIG["ubuntu"]["codename"]:
                return UbuntuVersionHealthCheckWarning(message=f"Ubuntu codename is {state}, not {EXPECTED_CONFIG['ubuntu']['codename']}")
        return HealthyResult()


UBUNTU_VERSION_HEALTH_CHECK: Final[LeafHealthCheck] = _UbuntuVersionHealthCheck()


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class InfinibandStatusHealthCheckError(HealthCheckError):
    """
    Raised when Infiniband is not fully active.
    """

    suggested_remediation: str = "\n".join(
        (
            "Infiniband is not active on all 8 devices, cleaning and reseating the transceivers often helps.",
        )
    )


@attr.s(auto_attribs=True, frozen=True)
class _InfinibandStatusHealthCheck(LeafHealthCheck):
    """
    Validates that all Infiniband devices are active.
    """

    def create_command(self) -> str:
        expected_network_type = EXPECTED_CONFIG['infiniband_status']['network_type']
        return f"ibstatus | grep -B 2 {expected_network_type} | grep -c ACTIVE"

    def validate_result(self, output: str, returncode: int) -> HealthCheckOutcome:
        if returncode != 0:
            expected_network_type = EXPECTED_CONFIG['infiniband_status']['network_type']
            return InfinibandStatusHealthCheckError(
                message=f"`ibstatus | grep -B 2 {expected_network_type} | grep -c ACTIVE` command failed with returncode {returncode} and output {output}"
            )
        output_stripped = output.strip()
        expected_active_devices = EXPECTED_CONFIG["infiniband_status"]["active_devices"]
        if output_stripped != expected_active_devices:
            return InfinibandStatusHealthCheckError(message=f"Expected {expected_active_devices} active devices. Got {output_stripped}")
        return HealthyResult()


IB_STATUS_HEALTH_CHECK: Final[LeafHealthCheck] = _InfinibandStatusHealthCheck()


@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class DmesgHealthCheckWarning(HealthCheckWarning):
    """Raised when there are unrecognized messages in dmesg"""

    suggested_remediation: str = "\n".join(
        (
            "Check on the unrecognized messages in dmesg, boot ends ~line 10000",
            "Investigate the underlying issue, if these messages don't indicate a problem, whitelist them",
        )
    )


DMESG_WHITELISTED_MESSAGES = WHITELISTED_MESSAGES
DMESG_WHITELISTED_REGEXES = [re.compile(regex) for regex in WHITELISTED_REGEX_STR]
DMESG_WHITELISTED_MESSAGE_RANGES = WHITELISTED_MESSAGE_RANGES


@attr.s(auto_attribs=True, frozen=True)
class _DmesgHealthCheck(LeafHealthCheck):
    def create_command(self) -> str:
        command_parts = (
            "sudo cat /var/log/dmesg && sudo dmesg",
            r"sed -E 's/^\[[0-9\.[:space:]]+\][[:space:]]*(kernel:)?[[:space:]]+//g'",
            r"sed -E 's/XE9680\/[A-Z0-9]{6}/SYSTEM_TAG/g'",
            "sed -E 's/0x[a-f0-9]+/HEX/g'",
            r"sed -E 's/audit\([0-9.:]+\)/TIME/g'",
            "sed -E 's/[a-f0-9]{8}(-[a-f0-9]{4}){3}-[a-f0-9]{12}/UUID/g'",
            "sed -E 's/pid[=[:space:]][0-9]+/PID/g'",
            "sed -E 's/[0-9]+ns/NS_TIME/g'",
            r"sed -E 's/[0-9]+[[:space:]]MB\/s/MB_S/g'",
            "sed -E 's/nvme[0-9]/NVME/g'",
            "sed -E 's/[0-9]{5,}[kK]/NUM_KB/g'",
            "sed -E 's/irq [0-9]+/irq/gI'",
            "sed -E 's/sg[0-9]+/sg/gI'",
            "sed -E 's/MAC address ([0-9a-f]{2}:){5}[0-9a-f]{2}/MAC/gI'",
            r"sed -E 's/0{4}(:[a-f0-9]{2}){2}\.[0-9a-f]/ADDR/gI'",
            r"sed -E 's/\[[:a-f0-9]{9}\]/ADDR/gI'",
            "sed -E 's/mem[0-9]+/MEM_ADDR/g'",
            "sed -E 's/CPU[0-9]+/CPU_NUM/g'",
            "sed -E 's/[[:space:]]([0-9]+:){4}[[:space:]]/ID/g'",
            "sed -E 's/[^0-9]([0-9]{2}:){2}[0-9]{2}[^0-9]/TIME/g'",
            "sed -E 's/ib[0-9]/IB_NUM/g'",
            "sed -E 's/host[0-9]+/HOST_NUM/g'",
            "sed -E 's/device-([0-9]+:){2}/DEVICE/g'",
            "sed -E 's/[0-9]{4}-[0-9]{2}-[0-9]{2}/DATE/g'",
            "sed -E 's/veth[0-9a-f]+/veth/g'",
            "sed -E 's/br-[0-9a-f]+/br/g'",
            'grep -v -E "^[[:space:]]*$"',
            "cat -n",
            "sort -k2",
        )
        command = " | ".join(command_parts)
        return command

    def validate_result(self, output: str, returncode: int) -> HealthCheckOutcome:
        if returncode != 0:
            return DmesgHealthCheckWarning(message=f"The `dmesg` command failed with return code {returncode}.")
        lines = output.splitlines()
        lines = sorted(lines, key=lambda line: line.split("\t", 1)[0])
        unrecognized_lines = []
        is_ignoring_until_end_marker = None

        for full_line in lines:
            full_line = full_line.strip()
            if len(full_line.split("\t")) == 1:
                continue
            line_num_str, line = full_line.split("\t", 1)
            if len(line.split()) < 2:
                continue  # Skip lines that are probably cut off
            if any(message in line for message in DMESG_WHITELISTED_MESSAGES):
                continue
            if any(regex.match(line) for regex in DMESG_WHITELISTED_REGEXES):
                continue

            if line_num_str.strip().isdigit():
                # search in line for messages which indicate we need to ignore a range
                for message, end_marker in DMESG_WHITELISTED_MESSAGE_RANGES.items():
                    if message in line:
                        is_ignoring_until_end_marker = end_marker
                        break

                if is_ignoring_until_end_marker is not None:
                    # search in line for messages which indicate we should now stop ignoring
                    if is_ignoring_until_end_marker in line:
                        is_ignoring_until_end_marker = None
                    continue

            unrecognized_lines.append(f"{line_num_str}: {line}")

        total_unrecognized_lines = len(unrecognized_lines)
        if total_unrecognized_lines > 0:
            if total_unrecognized_lines > 5:
                unrecognized_lines = unrecognized_lines[:5] + [f"...{total_unrecognized_lines - 5} more..."]
            error_message = "\n".join(
                [f"{total_unrecognized_lines} unrecognized dmesg line(s) detected"] + unrecognized_lines
            )
            return DmesgHealthCheckWarning(
                message=error_message,
            )
        return HealthyResult()


DMESG_WHITELIST_HEALTH_CHECK: Final[LeafHealthCheck] = _DmesgHealthCheck()


def get_health_check_from_str(health_checks: str) -> Optional[HealthCheck]:
    health_checks_to_run = []
    health_checks_raw = health_checks.split(",")
    health_check_classes = LeafHealthCheck.__subclasses__() + CompoundHealthCheck.__subclasses__()
    for health_check_class in health_check_classes:
        health_check_name = health_check_class.__name__.lower().replace("_", "")
        if health_check_name in [health_check_raw.lower().replace("_", "") for health_check_raw in health_checks_raw]:
            health_checks_to_run.append(health_check_class())  # type: ignore
    if len(health_checks_to_run) == 1:
        return health_checks_to_run[0]
    if len(health_checks_to_run) > 1:
        return CompoundHealthCheck(tuple(health_checks_to_run))
    return None


# A health check that checks the health of the GPUs.
GPU_HEALTH_CHECK: Final[HealthCheck] = CompoundHealthCheck(
    (
        NVIDIA_SMI_HEALTH_CHECK,
        DMESG_XID_HEALTH_CHECK,
    )
)

ALL_HEALTH_CHECKS: Final[HealthCheck] = CompoundHealthCheck(
    (
        DMESG_WHITELIST_HEALTH_CHECK,
        IB_HEALTH_CHECK,
        NVLINK_HEALTH_CHECK,
        NVLINK_STATUS_HEALTH_CHECK,
        ANSIBLE_FACT_GATHERING_HANG_HEALTH_CHECK,
        GPU_HEALTH_CHECK,
        DMESG_PCIE_HEALTH_CHECK,
        DOCKER_NVIDIA_COMMUNICATION_HEALTH_CHECK,
        DOCKER_INFO_HEALTH_CHECK,
        ZPOOL_HEALTH_CHECK,
        GDR_ENABLED_HEALTH_CHECK,
        FLINT_VERSION_HEALTH_CHECK,
        UBUNTU_VERSION_HEALTH_CHECK,
        VBIOS_VERSION_HEALTH_CHECK,
        IB_STATUS_HEALTH_CHECK,
        DMESG_NVLINK_HEALTH_CHECK,
        SWAP_HEALTH_CHECK,
        DISK_SPACE_HEALTH_CHECK,
    )
)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
