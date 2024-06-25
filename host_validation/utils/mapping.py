from abc import ABC
from abc import abstractmethod
from copy import deepcopy
from functools import cached_property
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import Self
from typing import Tuple
from typing import TypeVar

T = TypeVar("T")

TV = TypeVar("TV")


def remove_none(data: Iterable[Optional[T]]) -> List[T]:
    return [x for x in data if x is not None]


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


def freeze_mapping(*args: object, **kwargs: object) -> _FrozenDict:
    return _FrozenDict(*args, **kwargs)
