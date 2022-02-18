import numpy as np
import matplotlib.pyplot as plt
from functions import *

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
    # plt.savefig('pixel_plot.png')

    plt.show()


def add_noise_to_pattern(pattern, perc_noise):
    pattern_with_noise = np.copy(pattern)
    len_pattern = pattern_with_noise.shape[0]
    index = np.random.choice(len_pattern, int(len_pattern * perc_noise), replace=False)
    pattern_with_noise[index] = pattern_with_noise[index] * -1
    return pattern_with_noise


def main():
    np.random.seed(2)
    # patterns = np.vstack((x1, x2, x3))
    # w = patterns.T @ patterns
    # x = synch_update(x3d, w)

    pic_data = np.genfromtxt('pict.dat', delimiter=',')

    patterns = pic_data.reshape((-1, 1024))

    # for this part only store first 3 patterns
    train_patterns = patterns[:3, :]
    w = train_patterns.T @ train_patterns

    noise_range = [0.45] #np.linspace(1, 10, 10) * 0.1

    # do also n trials without seed
    for train_pattern in train_patterns:
        for noise_perc in noise_range:
            pattern_with_noise = add_noise_to_pattern(train_pattern, noise_perc)

            print_pattern(train_pattern)
            print_pattern(pattern_with_noise)
            first_synch = synch_update(pattern_with_noise, w)
            print_pattern(first_synch)
            second_synch = synch_update(first_synch, w)
            print_pattern(second_synch)

            #print_pattern(asynch_update(pattern_with_noise, w, plot=False, energy=False))


if __name__ == "__main__":
    main()
