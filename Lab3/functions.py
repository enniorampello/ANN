import matplotlib.pyplot as plt
import numpy as np


def synch_update(x_input, w, plot=False):
    old_input = x_input
    new_input = np.sign(old_input @ w)
    while (old_input != new_input).any():
        old_input = new_input
        new_input = np.array([1 if x >= 0 else -1 for x in old_input @ w])

    if plot:
        print_pattern(new_input)
    return new_input

def asynch_update(x_input, w, plot=False, energy=False):
    all_pixels = np.arange(x_input.shape[0])
    old_input = x_input
    new_input = old_input

    it = 1
    energies = []
    while True:
        np.random.shuffle(all_pixels)

        for pixel in all_pixels:
            new_input[pixel] = 1 if new_input @ w[pixel] >= 0 else -1
            energies.append(get_energy(new_input, w))
            if it % 100 == 0 and plot:
                print_pattern(new_input, it=it)
            it += 1

        if (old_input == new_input).all():
            if plot:
                print_pattern(new_input, it=it)
            if energy:
                energy_plot(energies)
            return new_input

        old_input = new_input

def print_pattern(pattern, it=None):
    pattern = pattern.reshape((32, 32))

    plt.figure()
    title = "pixel_plot"
    if it is not None:
        title += f' - it: {it}'

    plt.title(title)
    plt.imshow(pattern)
    # save a plot
    #plt.savefig('pixel_plot.png')
    plt.show()

def energy_plot(energies):
    plt.figure()
    plt.xlabel('Iteration')
    plt.ylabel('Energy')
    plt.title('Asynchronous update\nEnergy associated with pattern after every iteration')
    plt.plot(np.arange(1, len(energies) + 1), energies)
    plt.show()

def get_energy(pattern, w):
    return - w @ pattern @ pattern

def gen_random_weights(patterns_shape):
    return np.random.normal(0, scale=1, size=(patterns_shape, patterns_shape))