# Utilities for functions in file file_loss.py
from operator import mul
from functools import reduce

def product(l):
    """
      Implement the product of elements in a list.
    
    Args:
        l (list): List of elements we want to multiply.

    Returns: 
           The product of elements in the list l.
    """
    return reduce(mul, l, 1)
