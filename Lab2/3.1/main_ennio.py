import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from functions import *

LR = 0.001
NUM_NODES = 7
MAX_EPOCHS = 500
SIGMA = 1

SINE = True

np.random.seed(3)


def main():
    if SINE:
        targets = sin()
    else:
        targets = square()
    patterns = np.linspace(0, 2*np.pi, int(2*np.pi/0.1)).reshape(int(2*np.pi/0.1), 1)
    mu = init_means()
    w = init_weights()
    pred = [forward_pass(x, mu, w)[1] for x in patterns]
    
    phi_mat = np.zeros((NUM_NODES, patterns.shape[0]))
    for i in range(NUM_NODES):
        for j in range(patterns.shape[0]):
            phi_mat[i][j] = phi(abs(mu[i] - patterns[j]))

    w = np.linalg.inv(phi_mat @ phi_mat.T) @ phi_mat @ targets
    pred = [forward_pass(x, mu, w)[1] for x in patterns]

    plt.figure()
    plt.plot(patterns, pred)
    plt.show()


if __name__ == '__main__':
    main()