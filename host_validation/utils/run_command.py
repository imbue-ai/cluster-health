import subprocess

IP = str
import shlex
from typing import Protocol

import attr


def run_local_command(
    command: str,
) -> None:
    # This call to subprocess.Popen is not robust and is meant to be a placeholder for whatever method
    # you use for running arbitrary commands locally.
    process = subprocess.Popen(
        command.split(" "),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    stdout, stderr = process.communicate()
    return


@attr.s(auto_attribs=True, frozen=True)
class ProcessResult:
    returncode: int
    output: str


class CommandRunner(Protocol):
    def run_command(self, command: str, **kwargs: object) -> ProcessResult:
        ...

    @property
    def ip(self) -> IP:
        ...


@attr.s(auto_attribs=True, frozen=True)
class ContainerSSHConnectionData:
    ip: str
    port: int
    user: str

    def run_command(self, command: str) -> None:
        escaped_command = shlex.quote(command)
        run_local_command(f"ssh {self.user}@{self.ip} -p {self.port} {escaped_command}")


@attr.s(auto_attribs=True, frozen=True)
class RemoteCommandRunner(CommandRunner):
    connection: ContainerSSHConnectionData

    def run_command(self, command: str, **kwargs: object) -> ProcessResult:
        # This is a placeholder for whatever method you use to run commands over ssh
        self.connection.run_command(command, is_checked=True)
        return ProcessResult(returncode=0, output=str())

    @property
    def ip(self) -> IP:
        return self.connection.ip

    def __str__(self) -> str:
        return f"{self.connection.ip}:{self.connection.port}"


@attr.s(auto_attribs=True, frozen=True)
class FullConnection:
    ssh_connection: CommandRunner
    internal_ip: str
