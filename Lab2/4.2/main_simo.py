from functions import *

MAX_EPOCHS = 1000
LR = 0.2

np.random.seed(6)

def get_all_distances(cities, output_nodes):
    # dists = np.zeros(shape=(cities.shape[0], output_nodes.shape[0]))
    dists = [[[]for _ in range(output_nodes.shape[0])] for _ in range(cities.shape[0])]
    for idx_city in range(cities.shape[0]):
        for idx_output in range(output_nodes.shape[0]):
            dists[idx_city][idx_output] = (euclidean_distance(cities[idx_city], output_nodes[idx_output]), idx_output)

    return dists

def main():
    cities = get_city_matrix()
    w = init_weights((10, 2))

    for epoch in range(MAX_EPOCHS):
        neigh_size = 2 - 2 / MAX_EPOCHS * epoch
        print(f'EPOCH {epoch} neigh_size {round(neigh_size)}')
        for city in cities:
            w, _ = update_weights(city, w, neigh_size, LR, epoch)

    all_distances = get_all_distances(cities, w)

    for i in range(len(all_distances)):
        all_distances[i] = sorted(all_distances[i], key=lambda x: x[0])

    cities_to_assign = [x for x in range(cities.shape[0])]



    # result = {}
    # for i in range(cities.shape[0]):
    #     for pos in range(10):
    #         winning_idx = get_node(cities[i, :], w, pos)
    #         if winning_idx not in result.keys():
    #             result[winning_idx] = cities[i, :]
    #             break
    # plot_path(result)


if __name__ == '__main__':
    main()