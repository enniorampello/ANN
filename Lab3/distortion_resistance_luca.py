from functions import *

N_TRIALS = 1


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


def get_perc_of_recovered_pixels(pattern, updated_pattern):
    return np.sum(pattern == updated_pattern) / pattern.shape[0]


def main():
    # patterns = np.vstack((x1, x2, x3))
    # w = patterns.T @ patterns
    # x = synch_update(x3d, w)

    pic_data = np.genfromtxt('pict.dat', delimiter=',')

    patterns = pic_data.reshape((-1, 1024))

    # for this part only store first 3 patterns
    train_patterns = patterns[:3, :]
    w = train_patterns.T @ train_patterns

    noise_range = np.linspace(1, 10, 100) * 0.1
    perc_recovered_pixels_dict = {}
    for i, train_pattern in enumerate(train_patterns):
        perc_recovered_pixels_dict[i] = []
        print("_" * 80)
        print("pattern " + str(i))
        for noise_perc in noise_range:
            recovered_pixels_of_pattern = []
            for trial_idx in range(N_TRIALS):
                pattern_with_noise = add_noise_to_pattern(train_pattern, noise_perc)

                #print_pattern(train_pattern)
                #print_pattern(pattern_with_noise)
                updated_pattern = synch_update(pattern_with_noise, w)
                #print_pattern(updated_pattern)
                recovered_pixels_of_pattern.append(get_perc_of_recovered_pixels(train_pattern, updated_pattern))

            perc_recovered_pixels_dict[i].append(np.mean(recovered_pixels_of_pattern))

        plt.plot(noise_range, perc_recovered_pixels_dict[i], label="pattern " + str(i))
    plt.title("Percentage of recovered units with increasing percentage of flipped units")
    plt.xlabel("Percentage of noise (flipped units)")
    plt.ylabel("Percentage of recovered units")
    plt.legend()
    plt.show()

    print(perc_recovered_pixels_dict)


if __name__ == "__main__":
    main()
