from functions import *
import numpy as np
import itertools
import matplotlib.pyplot as plt

def synch_update(x_input, w):
    old_input = x_input
    new_input = np.sign(old_input @ w)
    while (old_input != new_input).any():
        old_input = new_input
        new_input = np.sign(old_input @ w)
    return new_input

def print_pattern(pattern):
    pattern = pattern.reshape((32, 32))

    # creating a plot
    plt.figure()

    # customizing plot
    plt.title("pixel_plot")
    plt.imshow(pattern)

    # save a plot
    #plt.savefig('pixel_plot.png')

    plt.show()



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


# for this part only store first 3 patterns
patterns = patterns[:3, :]
w = patterns.T @ patterns

# check that they are all stable points
for pattern in patterns:
    print((pattern == np.sign(pattern @ w)).all())

print_pattern(p11)
print_pattern(synch_update(p11, w))