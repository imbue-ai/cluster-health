import os
import shlex
import subprocess
import tempfile
import time
from abc import ABC
from abc import abstractmethod
from collections.abc import Mapping
from contextlib import contextmanager
from copy import deepcopy
from functools import cached_property
from itertools import groupby
from threading import Event
from threading import Thread
from typing_extensions import Any
from typing_extensions import Callable
from typing_extensions import Dict
from typing_extensions import Iterable
from typing_extensions import Iterator
from typing_extensions import List
from typing_extensions import NoReturn
from typing_extensions import Optional
from typing_extensions import Protocol
from typing_extensions import Self
from typing_extensions import Tuple
from typing_extensions import TypeVar
from typing_extensions import overload
from uuid import uuid4

import attr


class _SupportsLessThan(Protocol):
    def __lt__(self, __other: Any) -> bool:
        ...


T = TypeVar("T")

TK = TypeVar("TK", bound=_SupportsLessThan)
TV = TypeVar("TV")


def group_by_helper(data: Iterable[TV], get_key: Callable[[TV], TK]) -> Dict[TK, List[TV]]:
    data = sorted(data, key=get_key)
    return {k: list(g) for k, g in groupby(data, get_key)}


class FrozenMapping(Mapping[T, TV], ABC):
    @abstractmethod
    def __hash__(self) -> int:
        ...


# NOTE: `_key` is not `sorted` because A. not all python objects are sortable and python dictionaries are insertion-ordered.
class _FrozenDict(Dict[T, TV], FrozenMapping[T, TV]):
    def _key(self) -> Tuple[Tuple[T, TV], ...]:
        return tuple(self.items())

    @cached_property
    def _hash(self) -> int:
        return hash(self._key())

    def __hash__(self) -> int:  # type: ignore
        return self._hash

    def _mutation_error(self, method: str) -> RuntimeError:
        return RuntimeError(f"Cannot call mutation method {method} on _FrozenDict {self}")

    def __setitem__(self, __name: T, __value: TV) -> NoReturn:
        raise self._mutation_error("__setitem__")

    def __delitem__(self, __name: T) -> NoReturn:
        raise self._mutation_error("__delitem__")

    def update(self, __m: Mapping[T, TV]) -> NoReturn:  # type: ignore
        raise self._mutation_error("update")

    def setdefault(self, __name: T, __value: TV) -> NoReturn:
        raise self._mutation_error("setdefault")

    def pop(self, __name: T, __default: TV) -> NoReturn:  # type: ignore
        raise self._mutation_error("pop")

    def popitem(self) -> NoReturn:
        raise self._mutation_error("popitem")

    def clear(self) -> NoReturn:
        raise self._mutation_error("clear")

    def __repr__(self) -> str:
        return f"_FrozenDict({super().__repr__()})"

    def __copy__(self) -> Self:
        return type(self)(self)

    def __deepcopy__(self, memo: Dict[int, Any]) -> Self:
        memo[id(self)] = self
        copied_items = ((deepcopy(key, memo), deepcopy(value, memo)) for key, value in self.items())
        return type(self)(copied_items)

    def __reduce__(self) -> Tuple[Any, ...]:
        return (_FrozenDict, (dict(self),))


@overload
def freeze_mapping(**kwargs: TV) -> FrozenMapping[str, TV]:
    ...


@overload
def freeze_mapping(mapping: Mapping[T, TV], **kwargs: TV) -> FrozenMapping[T, TV]:
    ...


@overload
def freeze_mapping(__iterable: Iterable[Tuple[T, TV]]) -> FrozenMapping[T, TV]:
    ...


@overload
def freeze_mapping(__iterable: Iterable[Tuple[T, TV]], **kwargs: TV) -> FrozenMapping[T, TV]:
    ...


def freeze_mapping(*args: object, **kwargs: object) -> _FrozenDict:
    return _FrozenDict(*args, **kwargs)


def remove_none(data: Iterable[Optional[T]]) -> List[T]:
    return [x for x in data if x is not None]


SUBPROCESS_STOPPED_BY_REQUEST_EXIT_CODE = -9999


def _set_event_after_time(event: Event, seconds: float) -> None:
    event.wait(seconds)
    event.set()


@contextmanager
def get_expiration_event(seconds: float) -> Iterator[Event]:
    event = Event()
    # Since we set `daemon=True`, we don't need to join the thread.
    Thread(target=_set_event_after_time, args=(event, seconds), name=f"timeout_thread_{uuid4()}", daemon=True).start()
    try:
        yield event

    finally:
        event.set()


@attr.s(auto_exc=True, auto_attribs=True)
class CommandError(Exception):
    """
    An error that occurred while running a command.
    """

    command: str
    returncode: int
    output: str
    ssh_command: Optional[str] = None


@attr.s(auto_exc=True, auto_attribs=True)
class CompletedProcess:
    """
    Mostly a reimplementation of subprocess.CompletedProcess but allows us to deal with some specific concerns.
    A class to make process results easier to work with for us. We have a couple concerns that are different from typical:
    We run commands over SSH a lot and care about making sure that those errors clearly show both the command being run and the host being run on.
    We put the output from stdout and stderr together (should we!?).
    There's no support for binary commands.
    """

    returncode: int
    output: str
    command: str

    def check(self) -> Self:
        if self.returncode != 0:
            raise CommandError(
                command=self.command,
                returncode=self.returncode,
                output=self.output,
            )
        return self


@attr.s(auto_exc=True, auto_attribs=True)
class RemoteCompletedProcess(CompletedProcess):
    """
    A remote completed process. Beyond CompletedProcess, includes ssh information.
    """

    ssh_command: str

    def check(self) -> Self:
        if self.returncode != 0:
            raise CommandError(
                command=self.command,
                ssh_command=self.ssh_command,
                returncode=self.returncode,
                output=self.output,
            )
        return self


def run_local_command(
    command: str,
    is_checked: bool = True,
    timeout_sec: Optional[int] = None,
    shutdown_timeout_sec: int = 30,
) -> CompletedProcess:
    process = subprocess.Popen(
        command,
        shell=True,
        executable="/bin/bash",
        bufsize=1,
        encoding="UTF-8",
        errors="replace",
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**os.environ, "TERM": "dumb"},
    )
    exit_code = None
    start_time = time.time()
    while exit_code is None:
        if timeout_sec is not None and time.time() - start_time > timeout_sec:
            print("Terminating process due to timeout.")
            process.terminate()
            try:
                process.wait(timeout=shutdown_timeout_sec)
            except subprocess.TimeoutExpired as e:
                # this sends SIGKILL which immediately kills the process
                process.kill()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired as e:
                    print(f"process {process.pid} didn't terminate after kill")
            # use a special exit code so it doesn't look like this was a real failure of the process
            exit_code = SUBPROCESS_STOPPED_BY_REQUEST_EXIT_CODE
            break
        exit_code = process.poll()
        time.sleep(0.1)
    if exit_code == 0:
        output = process.stdout.read()
    else:
        output = "ERROR"
    result = CompletedProcess(
        returncode=exit_code,
        output=output,
        command=command,
    )
    if is_checked:
        result.check()

    return result


pipe_to_local_file_cmd_suffix = ""


def run_remote_command(
    machine_ssh_command: str,
    remote_command: str,
    is_checked: bool = True,
    timeout_sec: Optional[int] = None,
    shutdown_timeout_sec: int = 30,
) -> RemoteCompletedProcess:
    """
    :raises SSHConnectionError: if `is_checked and returncode == 255` (the ssh reserved error code)
    :raises RemoteCommandError: if `is_checked and returncode not in (0, 255)`
    """
    escaped_remote_command = shlex.quote(remote_command)
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file_name = temp_file.name
    command = f"{machine_ssh_command} {escaped_remote_command} 2>&1 | tee {temp_file_name} > /dev/null"
    result = run_local_command(
        command=command,
        is_checked=False,
        timeout_sec=timeout_sec,
        shutdown_timeout_sec=shutdown_timeout_sec,
    )

    remote_result = RemoteCompletedProcess(
        returncode=result.returncode,
        output=open(temp_file_name, "r").read(),
        ssh_command=machine_ssh_command,
        command=remote_command,
    )
    os.remove(temp_file_name)
    if is_checked:
        remote_result.check()
    return remote_result


DISABLE_HOST_KEY_CHECKING = f" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR "


@attr.s(auto_attribs=True, frozen=True)
class SSHConnectionData:
    ip: str
    port: int
    user: str

    def get_ssh_command(
        self,
        connection_timeout_seconds: Optional[int] = 10,
    ) -> str:
        connection_timeout = ""
        if connection_timeout_seconds is not None:
            connection_timeout = f"-o ConnectTimeout={connection_timeout_seconds}"

        base_command = f"ssh {DISABLE_HOST_KEY_CHECKING} -p {self.port}"
        return f"{base_command} {connection_timeout} {self.user}@{self.ip}"
