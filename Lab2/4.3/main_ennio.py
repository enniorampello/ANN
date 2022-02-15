import numpy as np
import fileinput
import matplotlib
import matplotlib.pyplot as plt
from functions import *
from tabulate import tabulate

MAX_EPOCHS = 50
LR = 0.2

VOTES_SHAPE = (349, 31)

# data paths
VOTES_PATH = "Lab2/4.1/data/votes.dat"
PARTIES_PATH = "Lab2/4.1/data/mpparty.dat"
GENDERS_PATH = "Lab2/4.1/data/mpsex.dat"
DISTRICTS_PATH = "Lab2/4.1/data/mpdistrict.dat"
NAMES_PATH = "Lab2/4.1/data/mpnames.txt"

parties_names = {-1: '', 0: 'np', 1: 'm', 2: 'fp', 3: 's', 4: 'v', 5: 'mp', 6: 'kd', 7: 'c'}


def init_weights(size=(100, 84)):
    return np.random.random(size=size)


def update_weights(pattern, w, neigh_size, lr, epoch):
    # assert neigh_size % 2 == 0, 'the size of the neighborhood should be an even number'
    min_dist = float("inf")
    winner_idx = -1
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            dist = euclidean_distance(pattern, w[i][j])
            if dist < min_dist:
                min_dist = dist
                winner_idx = (i, j)
    neighbors_idxs = get_neighbors_idxs(winner_idx, neigh_size, w.shape[0], w.shape[1])
    for idx in neighbors_idxs:
        w[idx[0]][idx[1]] += lr * (pattern - w[idx[0]][idx[1]])
    return w, winner_idx


def euclidean_distance(a, b):
    return np.linalg.norm(a - b)


def get_neighbors_idxs(winner_idx, neigh_size, num_nodes_x, num_nodes_y):
    x = winner_idx[0]
    y = winner_idx[1]
    idxs = []
    for i in range(num_nodes_x):
        for j in range(num_nodes_y):
            if abs(i - x) <= round(neigh_size) and abs(j - y) <= round(neigh_size):
                idxs.append((i, j))
    return idxs


def get_node(pattern, w):
    min_dist = float('inf')
    min_idx = (-1, -1)
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            dist = euclidean_distance(pattern, w[i][j])
            if dist < min_dist:
                min_dist = dist
                min_idx = (i, j)
    return min_idx
    

def most_common(lst):
    return max(set(lst), key=lst.count)


def main():
    # load data
    # 0 = no-vote; 1 = yes-vote; 0.5 = missing vote
    # each row -is one mp; each col is one vote
    votes_mps = get_votes(VOTES_PATH, VOTES_SHAPE)
    # each elem of parties, genders, names is the party, gender, name of the correspondent mp

    # 0=no party, 1='m', 2='fp', 3='s', 4='v', 5='mp', 6='kd', 7='c'
    parties = get_parties(path=PARTIES_PATH)
    # Male 0, Female 1
    genders = get_genders(path=GENDERS_PATH)
    districts = get_districts(path=DISTRICTS_PATH)
    names = get_names(path=NAMES_PATH)

    w = init_weights((10, 10, 31))

    for epoch in range(MAX_EPOCHS):
        neigh_size = 2 - 2/MAX_EPOCHS*epoch
        print(f'EPOCH {epoch} neigh_distance {round(neigh_size)}')
        for votes in votes_mps:
            w, _ = update_weights(votes, w, neigh_size, LR, epoch)
    
    grid = [[[] for _ in range(w.shape[1])] for _ in range(w.shape[0])]
    for idx, votes in enumerate(votes_mps):
        i, j = get_node(votes, w)
        grid[i][j].append(int(parties[idx]))
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            if len(grid[i][j]) > 0:
                grid[i][j] = most_common(grid[i][j])
            else:
                grid[i][j] = -1
    print(tabulate(grid, tablefmt='fancy_grid'))
    heatmap(grid)


if __name__ == '__main__':
    main()





