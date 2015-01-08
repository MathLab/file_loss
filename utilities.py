# Utilities for functions in file file_loss.py

from itertools import (cycle, islice, combinations, chain)
from operator import mul
from functools import reduce

def window(seq, n):
    """
      Implement a sliding window of size n for a sequence seq.

    Args: 
        seq (list): The sequence we want to apply the sliding window.
        n   (int): The size of the sliding window.

    Returns:
           Returns a sliding window as a tuple.
    """
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

def product(l):
    """
      Implement the product of elements in a list.
    
    Args:
        l (list): List of elements we want to multiply.

    Returns: 
           The product of elements in the list l.
    """
    return reduce(mul, l, 1)
