from functions import *
import numpy as np
import itertools

def synch_update(x_input, w):
    old_input = x_input
    new_input = np.sign(old_input @ w)
    while (old_input != new_input).any():
        old_input = new_input
        new_input = np.sign(old_input @ w)
    return new_input

x1 = np.array([-1, -1, 1, -1, 1, -1, -1, 1], ndmin=2)
x2 = np.array([-1, -1, -1, -1, -1, 1, -1, -1], ndmin=2)
x3 = np.array([-1, 1, 1, -1, -1, 1, -1, 1], ndmin=2)

x1d = np.array([1, -1, 1, -1, 1, -1, -1, 1], ndmin=2)
x2d = np.array([1, 1, -1, -1, -1, 1, -1, -1], ndmin=2)
x3d = np.array([1, 1, 1, -1, 1, 1, -1, 1], ndmin=2)


patterns = np.vstack((x1, x2, x3))
w = patterns.T @ patterns


x = synch_update(x3d, w)

combs = list(itertools.product([-1, 1], 8))
tot = 0
for c in combs:
    print(c)
    for perm in list(itertools.combinations_with_replacement(c,8)):
        pass
print(tot)

