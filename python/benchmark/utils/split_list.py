from itertools import islice
from typing import Iterable


def split_list(it: Iterable[int], s: int):
    """
    Split a list into sublists of size at most s.
    :param it: the iterator of the list to split
    :param s: the maximum size of the sublists
    """
    iterator = iter(it)
    while batch := list(islice(iterator, s)):
        yield batch
