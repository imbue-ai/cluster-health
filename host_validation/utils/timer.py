import time
from contextlib import contextmanager
from typing import Dict
from typing import Iterator
from typing import List
from typing import Mapping


class Timer(Mapping[str, float]):
    def __init__(self) -> None:
        self._times: Dict[str, List[float]] = {}

    @contextmanager
    def __call__(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            end = time.perf_counter()
            self._times.setdefault(name, []).append(1000 * (end - start))

    def __getitem__(self, name: str) -> float:
        if len(self._times[name]) == 1:
            return self._times[name][0]
        else:
            return max(self._times[name][1:])

    def __iter__(self) -> Iterator[str]:
        return iter(self._times)

    def __len__(self) -> int:
        return len(self._times)
