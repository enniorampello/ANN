import numpy as np

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
    

    
def get_neighbors_idxs(winner_idx, neigh_size, num_nodes):
    idxs = []
    for i in range(num_nodes):
        if abs(i - winner_idx) <= neigh_size/2 or \
            num_nodes - i + winner_idx <= neigh_size/2 or \
                num_nodes + 1 - winner_idx <= neigh_size/2:
            idxs.append(i)



def euclidean_distance(a, b):
    return np.linalg.norm(a - b)



def main():
    w = init_weights()



if __name__ == '__main__':
    main()