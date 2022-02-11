import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from functions import *

'''
1. Choose a number of nodes for the hidden layer.
2. Initialise the means of the Gaussians for the hidden nodes (mu).
3. Initialise the output weights. (normal distribution?) (w)
3. Start training the network:
    a. Feed the training set X into the network.
    b. For each node, compute phi(x(t) - w[i]).
    c. Compute the output as a weighted sum of the outputs of the hidden nodes.
'''

LR = 0.01
NUM_NODES = 9
MAX_EPOCHS = 500
SIGMA = 0.5

SINE = True

NOISE = True
SIGMA_NOISE = 0.1

BATCH = False
ES = True
PATIENCE = 10

PLOT = True


np.random.seed(5)

def main():

    patterns = np.linspace(0, 2 * np.pi, int(2 * np.pi / 0.1)).reshape(int(2 * np.pi / 0.1), 1)
    val_patterns = np.linspace(0.05, np.pi, int(np.pi / 0.1)).reshape(int(np.pi / 0.1), 1)
    test_patterns = np.linspace(np.pi + 0.05, 2 * np.pi, int(np.pi / 0.1)).reshape(int(np.pi / 0.1), 1)


    if SINE:
        targets = sin(patterns)
        val_targets = sin(val_patterns)
        test_targets = sin(test_patterns)
    else:
        targets = square(patterns)
        val_targets = square(val_patterns)
        test_targets = square(test_patterns)

    if NOISE:
        targets = add_noise(targets, SIGMA_NOISE)
        val_targets = add_noise(val_targets, SIGMA_NOISE)
        test_patterns = add_noise(test_patterns, SIGMA_NOISE)

    mu = init_means(NUM_NODES)
    w = init_weights(NUM_NODES)

    phi_mat = np.zeros((NUM_NODES, patterns.shape[0]))
    for i in range(NUM_NODES):
        for j in range(patterns.shape[0]):
            phi_mat[i][j] = phi(abs(mu[i] - patterns[j]), SIGMA)


    if BATCH:
        w = train_batch(phi_mat, targets)
    else:
        w = train_seq(patterns, targets, w, MAX_EPOCHS, SIGMA, mu, LR, PLOT, ES, val_patterns, val_targets, PATIENCE)



    if SINE:
        pred = [forward_pass(x, mu, w, SIGMA)[1] for x in patterns]
    else:
        pred = [1 if forward_pass(x, mu, w, SIGMA)[1] >= 0 else -1 for x in patterns]

    plt.figure()
    plt.plot(patterns, targets)
    plt.plot(patterns, pred)
    plt.show()


if __name__ == '__main__':
    main()