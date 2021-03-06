from functions import *
import itertools
import numpy as np
import pandas as pd
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


pic_data = np.genfromtxt('Lab3/pict.dat', delimiter=',')

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
STORED_PATTERNS = 0

RANDOM_PATTERNS = 300
N_DIMS = 100

BIASED_PATTERNS = False
SPARSE = True
ITERATIVE_W = True
NOISE_P = 0. # np.linspace(1, 10, 10) * 0.1
REMOVE_SELF = False

BIASES_SPARSE = np.linspace(0, 10, 10) * 0.1
ACTIVITIES = [0.1, 0.05, 0.01]


bias_act_max = pd.DataFrame(index=ACTIVITIES, columns=BIASES_SPARSE)
for BIAS_SPARSE in BIASES_SPARSE:
    for ACTIVITY in ACTIVITIES:
        patterns = patterns[:STORED_PATTERNS, :]
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
            max_stored = 0
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
                if c > max_stored:
                    max_stored = c
                print(f'{c} / {i+1} patterns are fixed points ')
            bias_act_max.loc[ACTIVITY, BIAS_SPARSE] = max_stored
            plt.plot(np.arange(1, len(stable_points) + 1), stable_points)
            # plt.show()

        else:
            w = patterns.T @ patterns
print(bias_act_max.to_latex())
    # random initialization
    # w = gen_random_weights(patterns.shape[1])
    # w = get_symmetric_weights(patterns.shape[1])

# check that they are all stable points
# for pattern in patterns:
#     print((pattern == np.sign(pattern @ w)).all())

# print_pattern(p11)

# x = synch_update(p10, w, plot=True)
# x = asynch_update(p10, w, plot=True, energy=False)

