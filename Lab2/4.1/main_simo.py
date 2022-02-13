import numpy as np
from functions import *

LR = 0.2


def init_weights(size=(100, 84)):
    return np.random.random(size=size)


def update_weights(pattern, w, neigh_size):
    assert neigh_size % 2 == 0, 'the size of the neighborhood should be an even number'
    min_dist = float("inf")
    idx = -1
    for i in range(w.shape(0)):
        dist = euclidean_distance(pattern, w[i])
        if dist < min_dist:
            min_dist = dist
            idx = i
    neighbors_idxs = get_neighbors_idxs(idx, neigh_size, w.shape[0])


def neighbourhood_kernel(t, winner, loser, sigma_0, tau):
    sigma = sigma_0 * np.exp(-t ** 2 / tau)
    h = np.exp(-(np.power(euclidean_distance(winner, loser), 2)) / (2 * np.power(sigma, 2)))
    return h


def learning_rate_decay(t, lr_0, tau):
    return lr_0 * np.exp(-t / tau)


def get_neighbors_idxs(winner_idx, neigh_size, num_nodes):
    idxs = []
    for i in range(num_nodes):
        if abs(i - winner_idx) <= neigh_size / 2 or \
                num_nodes - i + winner_idx <= neigh_size / 2 or \
                num_nodes + 1 - winner_idx <= neigh_size / 2:
            idxs.append(i)


def euclidean_distance(a, b):
    return np.linalg.norm(a - b)


def main():
    ANIMALS = False
    SALESMAN = True
    PARTY = False

    if ANIMALS:
        data = np.genfromtxt('data/animals.dat', delimiter=',')
        names = np.loadtxt('data/animalnames.txt', dtype=str)
        for i in range(len(names)):
            names[i] = names[i].replace("'", '')

        w = init_weights(size=(100, 84))

    if SALESMAN:
        data = get_city_matrix()

        w = init_weights(size=(10, 2))

    if PARTY:
        data = np.genfromtxt('data/votes.dat', delimiter=',')
        sex = np.genfromtxt('data/mpsex.dat', comments="%")
        party = np.genfromtxt('data/mpparty.dat', comments="%")
        districts = np.genfromtxt('data/mpdistrict.dat', comments="%")
        names = np.loadtxt('data/mpnames.txt', dtype=str, delimiter='\n')


        for i in range(len(names)):
            names[i] = names[i].replace("'", '')


if __name__ == '__main__':
    main()
