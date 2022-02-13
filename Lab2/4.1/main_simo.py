import numpy as np


def neighbourhood_kernel(t, winner, loser, sigma_0, tau):
    sigma = sigma_0 * np.exp(-t**2/tau)
    h = np.exp(-(np.power(euclidean_distance(winner, loser), 2))/(2 * np.power(sigma, 2)))
    return h


def learning_rate_decay(t, lr_0, tau):
    return lr_0 * np.exp(-t/tau)