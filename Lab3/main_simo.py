from functions import *
import numpy as np
import itertools
import matplotlib.pyplot as plt

def synch_update(x_input, w):
    old_input = x_input
    new_input = np.sign(old_input @ w)
    while (old_input != new_input).any():
        old_input = new_input
        new_input = np.array([1 if x >= 0 else -1 for x in old_input @ w])

    return new_input

def asynch_update(x_input, w, plot=False):
    all_pixels = np.arange(x_input.shape[0])
    old_input = x_input
    new_input = old_input

    it = 1
    while True:
        np.random.shuffle(all_pixels)

        for pixel in all_pixels:
            new_input[pixel] = 1 if new_input @ w[pixel] >= 0 else -1

            if it % 100 == 0:
                print_pattern(new_input, it=it)
            it += 1

        if (old_input == new_input).all():
            print_pattern(new_input, it=it)
            return new_input



        old_input = new_input



def print_pattern(pattern, it=None):
    pattern = pattern.reshape((32, 32))

    # creating a plot
    plt.figure()

    # customizing plot
    title = "pixel_plot"
    if it is not None:
        title += f' - it: {it}'

    plt.title(title)
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
# for pattern in patterns:
#     print((pattern == np.sign(pattern @ w)).all())

# print_pattern(p11)
# print_pattern(synch_update(p11, w))

x = asynch_update(p11, w, plot=True)
