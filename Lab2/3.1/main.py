import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from functions import *
from functions_MLP import *

'''
1. Choose a number of nodes for the hidden layer.
2. Initialise the means of the Gaussians for the hidden nodes (mu).
3. Initialise the output weights. (normal distribution?) (w)
3. Start training the network:
    a. Feed the training set X into the network.
    b. For each node, compute phi(x(t) - w[i]).
    c. Compute the output as a weighted sum of the outputs of the hidden nodes.
'''

LR = 0.001
NUM_NODES = 20
MAX_EPOCHS = 100
SIGMA = 0.5

SINE = True

NOISE = False
SIGMA_NOISE = 0.1

BATCH = False
ES = False
PATIENCE = 50

PLOT = True

# MLP params
MLP_ = False

# competitive learning constants
MAX_EPOCHS_CL = 10
COMPETITIVE = True
# strategy to avoid dead units
MORE_THAN_ONE_WINNER = True
NUM_OF_WINNERS = int(NUM_NODES / 3)
# learning rate for competitive learning part
LR_CL = 0.2


np.random.seed(5)

def main():

    patterns = np.linspace(0, 2 * np.pi, int(2 * np.pi / 0.1)).reshape(int(2 * np.pi / 0.1), 1)
    val_patterns = np.linspace(0.05, np.pi, int(np.pi / 0.1)).reshape(int(np.pi / 0.1), 1)
    test_patterns = np.linspace(np.pi + 0.05, 2 * np.pi, int(np.pi / 0.1)).reshape(int(np.pi / 0.1), 1)

    if SINE:
        # sine function
        targets = sin(patterns)
        val_targets = sin(val_patterns)
        test_targets = sin(test_patterns)
    else:
        # square function
        targets = square(patterns)
        val_targets = square(val_patterns)
        test_targets = square(test_patterns)

    if NOISE:
        targets = add_noise(targets, SIGMA_NOISE)
        val_targets = add_noise(val_targets, SIGMA_NOISE)
        test_patterns = add_noise(test_patterns, SIGMA_NOISE)

    mu = init_means(NUM_NODES)
    w = init_weights(NUM_NODES)

    phi_mat = np.zeros((NUM_NODES, patterns.shape[0]))
    for i in range(NUM_NODES):
        for j in range(patterns.shape[0]):
            phi_mat[i][j] = phi(abs(mu[i] - patterns[j]), SIGMA)

    if MLP_:
        v_MLP, w_MLP, preds = MLP(np.transpose(patterns), targets.reshape(targets.shape[0]),
                                  np.transpose(val_patterns), val_targets.reshape(val_targets.shape[0]),
                                  MAX_EPOCHS, NUM_NODES, LR, ES, PATIENCE)
        if PLOT:
            plot(patterns, targets, np.transpose(preds), LR, NUM_NODES, MAX_EPOCHS,
                 MLP=True, es=ES, patience=PATIENCE)

    if BATCH:
        w = train_batch(phi_mat, targets)
    else:
        # sequential - on-line
        if COMPETITIVE:
            # competitive learning
            competitive_learning(patterns, mu, LR_CL, NUM_NODES, MORE_THAN_ONE_WINNER, NUM_OF_WINNERS, MAX_EPOCHS_CL)

        w = train_seq(patterns, targets, w, MAX_EPOCHS,
                      SIGMA, mu, LR, PLOT, ES, val_patterns, val_targets, PATIENCE)

    if SINE:
        # sine function
        preds = [forward_pass(x, mu, w, SIGMA)[1] for x in patterns]
    else:
        # square function
        preds = [1 if forward_pass(x, mu, w, SIGMA)[1] >= 0 else -1 for x in patterns]

    if PLOT:
        plot(patterns, targets, preds, LR, NUM_NODES, MAX_EPOCHS,
            batch=BATCH, cl=COMPETITIVE, lr_cl=LR_CL, es=ES, patience=PATIENCE,
             epochs_cl=MAX_EPOCHS_CL, more_winners=MORE_THAN_ONE_WINNER)



if __name__ == '__main__':
    main()
