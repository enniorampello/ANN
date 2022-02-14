import numpy as np


LR = 0.2
MAX_EPOCHS = 20


def get_patterns():
    patterns_str = str(np.loadtxt('Lab2/4.1/data/animals.dat', dtype=str))
    patterns = np.array(patterns_str.split(','), dtype=int).reshape((32, 84))
    return patterns


def get_names():
    names = np.loadtxt('Lab2/4.1/data/animalnames.txt', dtype=str)
    for i in range(len(names)):
        names[i] = names[i].replace("'", '')
    return names
    

def init_weights(size=(100, 84)):
    return np.random.random(size=size)


def update_weights(pattern, w, neigh_size, lr, epoch):
    # assert neigh_size % 2 == 0, 'the size of the neighborhood should be an even number'
    min_dist = float("inf")
    idx = -1
    for i in range(w.shape[0]):
        dist = euclidean_distance(pattern, w[i])
        if dist < min_dist:
            min_dist = dist
            idx = i
    neighbors_idxs = get_neighbors_idxs(idx, neigh_size, w.shape[0])
    for i in neighbors_idxs:
        w[i] += lr * (pattern - w[i])
    return w, idx


def neighbourhood_kernel(t, winner, loser, sigma_0=1, tau=1):
    sigma = sigma_0 * np.exp(-t**2/tau)
    h = np.exp(-(np.power(euclidean_distance(winner, loser), 2))/(2 * np.power(sigma, 2)))
    return h


def learning_rate_decay(t, lr_0, tau):
    return lr_0 * np.exp(-t/tau)

    
def get_neighbors_idxs(winner_idx, neigh_size, num_nodes):
    idxs = []
    for i in range(num_nodes):
        if abs(i - winner_idx) <= int(neigh_size/2): # or \
            # num_nodes - i + winner_idx <= int(neigh_size/2) or \
            #     num_nodes + 1 - winner_idx <= int(neigh_size/2):
            idxs.append(i)
    return idxs


def euclidean_distance(a, b):
    return np.linalg.norm(a - b)


def main():
    patterns = get_patterns()
    names = get_names()
    w = init_weights()

    for epoch in range(MAX_EPOCHS):
        neigh_size = 50 - 50/MAX_EPOCHS*epoch
        print(f'EPOCH {epoch} neigh_size {neigh_size}')
        for pattern in patterns:
            w, _ = update_weights(pattern, w, neigh_size, LR, epoch)
    result = {}
    for i in range(patterns.shape[0]):
        _, winning_idx = update_weights(patterns[i], w, 2, LR, epoch)
        result[names[i]] = winning_idx
    result = dict(sorted(result.items(), key=lambda item: item[1]))
    print(result)



if __name__ == '__main__':
    main()