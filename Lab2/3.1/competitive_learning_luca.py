import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from functions import *

'''
1. Choose a number of nodes for the hidden layer.
2. Initialise the means of the Gaussians for the hidden nodes (mu).
3. Initialise the output weights. (normal distribution?) (w)
3. Start training the network:
    a. Feed the training set X into the network.
    b. For each node, compute phi(x(t) - w[i]).
    c. Compute the output as a weighted sum of the outputs of the hidden nodes.
'''

LR = 0.01
NUM_NODES = 9
MAX_EPOCHS = 1000
SIGMA = 0.5

SINE = True

NOISE = True
SIGMA_NOISE = 0.1

# competitive learning constants
MORE_THAN_ONE_WINNER = False
NUM_OF_WINNERS = int(NUM_NODES / 3)
# learning rate for competitive learning part
LR_CL = 0.2

np.random.seed(5)

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

def main():

    patterns = np.linspace(0, 2 * np.pi, int(2 * np.pi / 0.1)).reshape(int(2 * np.pi / 0.1), 1)

    if SINE:
        targets = sin(patterns)
    else:
        targets = square(patterns)

    if NOISE:
        targets = add_noise(targets, SIGMA_NOISE)

    mu = init_means(NUM_NODES)
    for epoch_idx in range(MAX_EPOCHS):
        # competitive learning
        patterns_idx = [i for i in range(patterns.shape[0])]
        # shuffle pattern indexes
        np.random.shuffle(patterns_idx)
        for i in range(patterns.shape[0]):
            # at each iteration of CL a training vector is randomly selected from the data
            selected_pattern = patterns[patterns_idx[i]]
            # find the closest RBF unit
            dist_from_rbf_nodes = []
            for node_idx in range(NUM_NODES):
                # not sure that the mus are the weight...
                dist_from_rbf_nodes.append(euclidean_distance(selected_pattern, mu[node_idx]))

            if MORE_THAN_ONE_WINNER:
                # n closest RBF units - winners
                nearest_rbf_nodes_idx = np.asarray(dist_from_rbf_nodes).argsort()[:NUM_OF_WINNERS]
                for winner_idx in nearest_rbf_nodes_idx:
                    # not sure
                    mu[winner_idx] += LR_CL * (selected_pattern - mu[winner_idx])
            else:
                # single closest RBF unit - winner
                nearest_rbf_node_idx = np.argmin(dist_from_rbf_nodes)
                # not sure
                mu[nearest_rbf_node_idx] += LR_CL * (selected_pattern - mu[nearest_rbf_node_idx])

    w = init_weights(NUM_NODES)
    pred = [forward_pass(x, mu, w, SIGMA)[1] for x in patterns]

    phi_mat = np.zeros((NUM_NODES, patterns.shape[0]))
    for i in range(NUM_NODES):
        for j in range(patterns.shape[0]):
            phi_mat[i][j] = phi(abs(mu[i] - patterns[j]), SIGMA)

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    line, = ax.plot(patterns, pred)
    fig.canvas.draw()
    plt.show(block=False)
    for epoch in range(MAX_EPOCHS):
        error = 0
        for pattern, target in zip(patterns, targets):
            h_out, _ = forward_pass(pattern, mu, w, SIGMA)
            w = update_weights(target, h_out, w, LR)
            error += abs(target - np.sum(h_out * w))
        error /= patterns.shape[0]
        print(f'EPOCH {epoch}\t| error {np.sum(error)}')

        if epoch % 10 == 0:
            #clear_output(wait=False)
            pred = [forward_pass(x, mu, w, SIGMA)[1] for x in patterns]
            #fig = plt.figure()
            line.set_xdata(patterns)
            line.set_ydata(pred)
            ax.relim()
            ax.autoscale_view(True,True,True)
            fig.canvas.draw()
            plt.pause(0.01)

    if SINE:
        pred = [forward_pass(x, mu, w, SIGMA)[1] for x in patterns]
    else:
        pred = [1 if forward_pass(x, mu, w, SIGMA)[1] >= 0 else -1 for x in patterns]

    plt.figure()
    plt.plot(patterns, targets)
    plt.plot(patterns, pred)
    plt.show()


if __name__ == '__main__':
    main()