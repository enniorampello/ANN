from functions import *
import itertools
import numpy as np
import itertools
import matplotlib.pyplot as plt


# x1 = np.array([-1, -1, 1, -1, 1, -1, -1, 1], ndmin=2)
# x2 = np.array([-1, -1, -1, -1, -1, 1, -1, -1], ndmin=2)
# x3 = np.array([-1, 1, 1, -1, -1, 1, -1, 1], ndmin=2)
#
# x1d = np.array([1, -1, 1, -1, 1, -1, -1, 1], ndmin=2)
# x2d = np.array([1, 1, -1, -1, -1, 1, -1, -1], ndmin=2)
# x3d = np.array([1, 1, 1, -1, 1, 1, -1, 1], ndmin=2)
#
#
# patterns = np.vstack((x1, x2, x3))
# w = patterns.T @ patterns
# x = synch_update(x3d, w)


pic_data = np.genfromtxt('pict.dat', delimiter=',')

patterns = pic_data.reshape((-1, 1024))

# distorted patterns
p10 = patterns[9, :]
p11 = patterns[10, :]
patterns = np.delete(patterns, (9, 10), 0)


# here we can choose how many patterns we want to store (least is 3)
STORED_PATTERNS = 0
RANDOM_PATTERNS = 300
N_DIMS = 100
ITERATIVE_W = True
NOISE_P = 0# np.linspace(1, 10, 10) * 0.1
REMOVE_SELF = True



patterns = patterns[:STORED_PATTERNS, :]

if RANDOM_PATTERNS > 0:
    for _ in range(RANDOM_PATTERNS):
        x = np.random.choice([-1, 1], size=(N_DIMS,))
        if patterns.shape[0] == 0:
            patterns = x
        else:
            patterns = np.vstack((patterns, x))


# get weight matrix iteratively and check if all patterns remain stable
if ITERATIVE_W:
    for i in range(patterns.shape[0]):
        w = get_weights(patterns[:i+1], REMOVE_SELF)
        c = 0
        for pattern in patterns[:i+1]:
            pattern = add_noise_to_pattern(pattern, NOISE_P)
            if (pattern == np.sign(pattern @ w)).all():
                c += 1
        print(f'{c} out of {i+1} patterns are fixed points ')
else:
    w = patterns.T @ patterns
    # random initialization
    # w = gen_random_weights(patterns.shape[1])
    # w = get_symmetric_weights(patterns.shape[1])

# check that they are all stable points
# for pattern in patterns:
#     print((pattern == np.sign(pattern @ w)).all())

# print_pattern(p11)

# x = synch_update(p11, w, plot=True)
# x = asynch_update(p11, w, plot=True, energy=True)

