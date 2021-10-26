from functools import reduce  # Required in Python 3
from typing import Iterable, TypeVar
import operator


T = TypeVar('T')
def prod(iterable: Iterable[T]) -> T:
    """
    Returns the product of the elements in the given iterable.
    """
    return reduce(operator.mul, iterable, 1)
