from typing import Tuple, TypeVar, Set, Any
T = TypeVar('T', int, float, Set[int])


def trd(x: Tuple[Any, Any, T]) -> T:
    return x[2]
