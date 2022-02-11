import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time

'''
1. Choose a number of nodes for the hidden layer.
2. Initialise the means of the Gaussians for the hidden nodes (mu).
3. Initialise the output weights. (normal distribution?) (w)
3. Start training the network:
    a. Feed the training set X into the network.
    b. For each node, compute phi(x(t) - w[i]).
    c. Compute the output as a weighted sum of the outputs of the hidden nodes.
'''

LR = 0.001
NUM_NODES = 7
MAX_EPOCHS = 500
SIGMA = 1

np.random.seed(3)

def sin(start=0, stop=2*np.pi):
    """

    Args:
        start (int, optional): starting point of the sequence. Set to 0 for training set and 0.05 for val and np.pi+0.5 for test. Defaults to 0.
        stop ([type], optional): ending point of the sequence. Only use with test/val. If val, set to np.pi, else 2*np.pi. Defaults to 2*np.pi.

    Returns:
        np.array: array containing the sequence of points
    """
    return np.sin(2*np.linspace(start, start+stop, int(stop/0.1))).reshape(int(stop/0.1), 1)

def square(start=0, stop=2*np.pi):
    return np.array([1 if np.sin(2*x) >= 0 else -1 for x in np.linspace(start, start+stop, int(stop/0.1))]).reshape(int(stop/0.1), 1)

def phi(r):
    return np.exp(-(r**2)/(2*SIGMA**2))

def init_means():
    return np.linspace(0, 2*np.pi, NUM_NODES).reshape(NUM_NODES, 1)

def init_weights():
    return np.random.normal(0, scale=1, size=(NUM_NODES, 1))

def forward_pass(pattern, mu, w):
    h_in = np.abs(mu - pattern)
    h_out = phi(h_in)
    o_out = np.sum(w * h_out)
    return h_out, o_out

def update_weights(target, h_out, w):
    w += LR * (target - np.sum(h_out * w)) * h_out
    return w

def print_function(f, start=0, stop=2*np.pi):
    x = np.linspace(start, start+stop, int(stop/0.1))
    plt.figure()
    plt.plot(x, f)
    plt.show()


def main():
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
    pred = [1 if forward_pass(x, mu, w)[1] >= 0 else -1 for x in patterns]

    plt.figure()
    plt.plot(patterns, square())
    plt.plot(patterns, pred)
    plt.show()





if __name__ == '__main__':
    main()