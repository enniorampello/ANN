import numpy as np
import matplotlib.pyplot as plt

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

def phi(r, sigma):
    return np.exp(-(r**2)/(2*sigma**2))

def init_means(num_nodes):
    return np.linspace(0, 2*np.pi, num_nodes).reshape(num_nodes, 1)

def init_weights(num_nodes):
    return np.random.normal(0, scale=1, size=(num_nodes, 1))

def forward_pass(pattern, mu, w, sigma):
    h_in = np.abs(mu - pattern)
    h_out = phi(h_in, sigma)
    o_out = np.sum(w * h_out)
    return h_out, o_out

def update_weights(target, h_out, w, lr):
    w += lr * (target - np.sum(h_out * w)) * h_out
    return w

def print_function(f, start=0, stop=2*np.pi):
    x = np.linspace(start, start+stop, int(stop/0.1))
    plt.figure()
    plt.plot(x, f)
    plt.show()