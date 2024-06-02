__all__ = ["weight_set_generator","generate_alternating_range","divide_by_gcd_matrix"]

from itertools import product
import numpy as np
import math

def weight_set_generator(kappa: int, n: int):
    a = list(product(range(-kappa, kappa + 1), repeat=n))
    for i in a:
        yield i

def generate_alternating_range(n,start_with_zero=True):
    if start_with_zero is True:
        sequence = [0]  # start with 0
    else:
        sequence = []
    
    for i in range(1, n + 1):
        sequence.append(i)
        sequence.append(-i)
    return sequence


def divide_by_gcd_matrix(A):
    n_rows, n_cols = A.shape
    B = np.zeros(A.shape)
    for i in range(n_cols):
        B[:,i] = - A[:,i] / math.gcd(*A[:,i])
    return B