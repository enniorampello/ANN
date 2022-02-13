from lib2to3.pygram import pattern_symbols
from tkinter.font import names
import numpy as np

LR = 0.2


def get_names():
    names = np.loadtxt('data/animalnames.txt', dtype=str)
    for i in range(len(names)):
        names[i] = names[i].replace("'", '')
    return names
    


def init_weights(size=(100, 84)):
    return np.random.random(size=size)


def update_weights(pattern, w, neigh_size, lr):
    assert neigh_size % 2 == 0, 'the size of the neighborhood should be an even number'
    min_dist = float("inf")
    idx = -1
    for i in range(w.shape(0)):
        dist = euclidean_distance(pattern, w[i])
        if dist < min_dist:
            min_dist = dist
            idx = i
    neighbors_idxs = get_neighbors_idxs(idx, neigh_size, w.shape[0])
    for i in neighbors_idxs:
        w[i] += lr * (pattern - w[i])
    return w

    
def get_neighbors_idxs(winner_idx, neigh_size, num_nodes):
    idxs = []
    for i in range(num_nodes):
        if abs(i - winner_idx) <= neigh_size/2 or \
            num_nodes - i + winner_idx <= neigh_size/2 or \
                num_nodes + 1 - winner_idx <= neigh_size/2:
            idxs.append(i)
    return idxs


def euclidean_distance(a, b):
    return np.linalg.norm(a - b)



def main():
    patterns = np.genfromtxt('data/animals.dat', delimiter=',')
    names = get_names()
    w = init_weights()


    for epoch in range(20):
        for pattern in patterns:
            pass



if __name__ == '__main__':
    main()