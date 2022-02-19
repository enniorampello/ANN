import matplotlib.pyplot as plt
import numpy as np

def get_weights(patterns, average_activity=None, remove_self=False, sparse=False):
    if sparse:
        w = (patterns - average_activity).T @ (patterns - average_activity)
    else:
        w = patterns.T @ patterns
    if remove_self:
        np.fill_diagonal(w, 0)
    return w / patterns.shape[1]

def synch_update(x_input, w, plot=False):
    old_input = x_input
    new_input = np.sign(old_input @ w)
    while (old_input != new_input).any():
        old_input = new_input
        new_input = np.array([1 if x >= 0 else -1 for x in old_input @ w])

    if plot:
        print_pattern(new_input)
    return new_input

def sparse_update(x_input, w, bias):
    return 0.5 + 0.5 * np.sign((x_input @ w) - bias)

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

def get_symmetric_weights(patterns_shape):
    w = np.random.normal(0, scale=1, size=(patterns_shape, patterns_shape))
    return 0.5 * (w + np.transpose(w))


def add_noise_to_pattern(pattern, perc_noise):
    pattern_with_noise = np.copy(pattern)
    len_pattern = pattern_with_noise.shape[0]
    index = np.random.choice(len_pattern, int(len_pattern * perc_noise), replace=False)
    pattern_with_noise[index] = pattern_with_noise[index] * -1
    return pattern_with_noise