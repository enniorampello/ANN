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
MAX_EPOCHS = 500
SIGMA = 0.5

SINE = True

NOISE = True
SIGMA_NOISE = 0.1


np.random.seed(5)

def main():

    patterns = np.linspace(0, 2 * np.pi, int(2 * np.pi / 0.1)).reshape(int(2 * np.pi / 0.1), 1)

    if SINE:
        targets = sin(patterns)
    else:
        targets = square(patterns)

    if NOISE:
        targets = add_noise(targets, SIGMA_NOISE)


    mu = init_means(NUM_NODES)
    w = init_weights(NUM_NODES)
    pred = [forward_pass(x, mu, w, SIGMA)[1] for x in patterns]

    phi_mat = np.zeros((NUM_NODES, patterns.shape[0]))
    for i in range(NUM_NODES):
        for j in range(patterns.shape[0]):
            phi_mat[i][j] = phi(abs(mu[i] - patterns[j]), SIGMA)

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)   
    line, = ax.plot(patterns, pred,,
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