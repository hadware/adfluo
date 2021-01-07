import logging
from typing import List

logger = logging.getLogger("mekhane")

def consecutive_couples(iterable):
    firsts = list(iterable)
    firsts.pop(-1)
    seconds = list(iterable)
    seconds.pop(0)
    for first, second in zip(firsts, seconds):
        yield first, second

def are_consecutive_int(l : List[int]):
    """Returns True if integers in the list are all consecutive"""
    return l == list(range(min(l), max(l) + 1))
