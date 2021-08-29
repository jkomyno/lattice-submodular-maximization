from typing import Callable, Tuple, Iterator, TypeVar

T = TypeVar('T')
V = TypeVar('V')

def map_fst(f: Callable[[T], V], it: Iterator[Tuple[T, ...]]) -> Iterator[Tuple[V, ...]]:
    """
    Map the first element of each tuple in the iterator.
    """
    def helper(t: Tuple[T, ...]) -> Tuple[V, ...]:
        fst, *rest = t
        return (f(fst), *rest)

    return map(helper, it)
