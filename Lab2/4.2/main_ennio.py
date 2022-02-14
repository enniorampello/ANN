import numpy as np
import fileinput
import matplotlib.pyplot as plt

MAX_EPOCHS = 200
LR = 0.2


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
        if abs(i - winner_idx) <= round(neigh_size) or \
            num_nodes - i + winner_idx <= round(neigh_size) or \
                num_nodes + 1 - winner_idx <= round(neigh_size):
            idxs.append(i)
    return idxs


def get_node(pattern, w, pos=0):
    distances = []
    for i in range(w.shape[0]):
        dist = euclidean_distance(pattern, w[i])
        distances.append([dist, i])
    distances = sorted(distances, key=lambda x: x[0])
    return distances[pos][1]
    
def get_distances(patterns, w):
    distances = np.empty((100, 3))
    i = 0
    for c in range(len(patterns)):
        for j in range(w.shape[0]):
            distances[i,0] = c
            distances[i,1] = j
            distances[i,2] = euclidean_distance(patterns[c], w[j])
            i += 1
    distances = distances[distances[:,2].argsort()]
    return distances


def plot_cities(cities):
    plt.figure()
    plt.title('Shortest path')
    plt.scatter(cities[:,0], cities[:,1])
    plt.show()


def plot_path(result, cities_dict):
    plt.figure()
    print(result)
    result[10] = result[0]
    print(result)
    length = 0
    for i in range(len(result)-1):
        length += euclidean_distance(cities_dict[result[i]], cities_dict[result[i+1]])
    print(length)
    x = [cities_dict[result[i]][0] for i in range(len(result))]
    y = [cities_dict[result[i]][1] for i in range(len(result))]
    plt.title(f'Salesman path. Length = {length:.2f}')
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
    cities_dict = {}
    for i in range(cities.shape[0]):
        cities_dict[i] = cities[i,:]
    print(cities_dict)
    distances = get_distances(cities_dict, w)

    result = {}
    for i in range(distances.shape[0]):
        if int(distances[i,0]) not in result.values() and int(distances[i,1]) not in result.keys():
            result[int(distances[i,1])] = int(distances[i,0])
    print(result)
    plot_path(result, cities_dict)


if __name__ == '__main__':
    main()