from functions import *
import itertools
import numpy as np
import itertools
import matplotlib.pyplot as plt


# x1 = np.array([-1, -1, 1, -1, 1, -1, -1, 1])
# x2 = np.array([-1, -1, -1, -1, -1, 1, -1, -1])
# x3 = np.array([-1, 1, 1, -1, -1, 1, -1, 1])
#
# x1d = np.array([1, -1, 1, -1, 1, -1, -1, 1])
# x2d = np.array([1, 1, -1, -1, -1, 1, -1, -1])
# x3d = np.array([1, 1, 1, -1, 1, 1, -1, 1])
# #
# #
# patterns = np.vstack((x1, x2, x3))
# w = patterns.T @ patterns
# x = synch_update(x2d, w)

pic_data = np.genfromtxt('pict.dat', delimiter=',')

patterns = pic_data.reshape((-1, 1024))

# distorted patterns
p10 = patterns[9, :]
p11 = patterns[10, :]
patterns = np.delete(patterns, (9, 10), 0)


# patterns[[3, 4]] = patterns[[4, 3]]

# for pattern in patterns:
#     print_pattern(pattern)

np.random.seed(0)
# patterns we want to store (least is 3)
STORED_PATTERNS = 4

RANDOM_PATTERNS = 0
N_DIMS = 100

BIASED_PATTERNS = False
SPARSE = False
ITERATIVE_W = True
NOISE_P = 0.1 # np.linspace(1, 10, 10) * 0.1
REMOVE_SELF = False
BIAS_SPARSE = 0.1, 0.2
ACTIVITY = 0.1

patterns = patterns[:STORED_PATTERNS, :]
print_pattern(add_noise_to_pattern(patterns[0], 0.8))
exit()

average_activity = None
if RANDOM_PATTERNS > 0:
    if BIASED_PATTERNS:
        patterns = np.sign(0.5 + np.random.normal(0, 1, size=(RANDOM_PATTERNS, N_DIMS)))
    elif SPARSE:
        for _ in range(RANDOM_PATTERNS):
            x = np.random.choice([0, 1], size=(N_DIMS,), p=[1-ACTIVITY, ACTIVITY])
            if patterns.shape[0] == 0:
                patterns = x
            else:
                patterns = np.vstack((patterns, x))
        average_activity = (1/(N_DIMS * RANDOM_PATTERNS)) * np.sum(patterns, axis=(0, 1))
    else:
        for _ in range(RANDOM_PATTERNS):
            x = np.random.choice([-1, 1], size=(N_DIMS,))
            if patterns.shape[0] == 0:
                patterns = add_noise_to_pattern(x, NOISE_P)
            else:
                new_p = add_noise_to_pattern(x, NOISE_P)
                patterns = np.vstack((patterns, new_p))


# get weight matrix iteratively and check if all patterns remain stable
if ITERATIVE_W:
    stable_points = []
    for i in range(patterns.shape[0]):
        w = get_weights(patterns[:i+1], average_activity, REMOVE_SELF, SPARSE)
        c = 0
        for pattern in patterns[:i+1]:
            # pattern = add_noise_to_pattern(pattern, NOISE_P)
            # if i+1== patterns.shape[0]:
            #     print_pattern(pattern)
            if SPARSE:
                if (pattern == sparse_update(pattern, w, BIAS_SPARSE)).all():
                    c += 1
            else:
                if (pattern == np.sign(pattern @ w)).all():
                    c += 1
        stable_points.append(c/(i+1))
        print(f'{c} / {i+1} patterns are fixed points ')
    plt.plot(np.arange(1, len(stable_points) + 1), stable_points)
    plt.show()
else:
    w = patterns.T @ patterns
    # random initialization
    # w = gen_random_weights(patterns.shape[1])
    # w = get_symmetric_weights(patterns.shape[1])

# check that they are all stable points
# for pattern in patterns:
#     print((pattern == np.sign(pattern @ w)).all())

# print_pattern(p11)

# x = synch_update(p10, w, plot=True)
# x = asynch_update(p10, w, plot=True, energy=False)

