from collections import deque
from typing import Iterator, TypeVar

T = TypeVar('T')


def get_last(it: Iterator[T]) -> T:
    """
    Return the last element of a given iterator.
    :param it: iterator to consume
    """
    dd = deque(it, maxlen=1)
    last = dd.pop()
    return last
