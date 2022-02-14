import numpy as np
import fileinput
import matplotlib.pyplot as plt

MAX_EPOCHS = 1000
LR = 0.2

np.random.seed(6)

def get_city_matrix(path='data/cities.dat', header_lines_to_skip=4):
    for line in fileinput.input(path, inplace=True):
        print('{}'.format(line.replace(';', '')), end='')
    city_matrix = np.genfromtxt(path, delimiter=',', skip_header=header_lines_to_skip)
    return city_matrix


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


def euclidean_distance(a, b):
    return np.linalg.norm(a - b)


def get_neighbors_idxs(winner_idx, neigh_size, num_nodes):
    idxs = []
    for i in range(num_nodes):
        if abs(i - winner_idx) <= round(neigh_size/2) or \
            num_nodes - i + winner_idx <= round(neigh_size/2) or \
                num_nodes + 1 - winner_idx <= round(neigh_size/2):
            idxs.append(i)
    return idxs


def get_node(pattern, w, pos=0):
    distances = []
    for i in range(w.shape[0]):
        dist = euclidean_distance(pattern, w[i])
        distances.append([dist, i])
    distances = sorted(distances, key=lambda x: x[0])
    return distances[pos][1]
    


def plot_cities(cities):
    plt.figure()
    plt.scatter(cities[:,0], cities[:,1])
    plt.show()


def plot_path(result):
    plt.figure()
    print(result)
    result[10] = result[0]
    print(result)
    length = 0
    for i in range(len(result)-1):
        length += euclidean_distance(result[i], result[i+1])
    print(length)
    x = [result[i][0] for i in range(len(result))]
    y = [result[i][1] for i in range(len(result))]
    plt.plot(x, y, 'ro-')
    plt.show()


def main():
    cities = get_city_matrix('Lab2/4.1/data/cities.dat')
    w = init_weights((10, 2))

    for epoch in range(MAX_EPOCHS):
        neigh_size = 2 - 2/MAX_EPOCHS*epoch
        print(f'EPOCH {epoch} neigh_size {round(neigh_size)}')
        for city in cities:
            w, _ = update_weights(city, w, neigh_size, LR, epoch)
    result = {}
    for i in range(cities.shape[0]):
        for pos in range(10):
            winning_idx = get_node(cities[i,:], w, pos)
            if winning_idx not in result.keys():
                result[winning_idx] = cities[i,:]
                break
    plot_path(result)


if __name__ == '__main__':
    main()