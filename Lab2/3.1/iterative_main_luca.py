import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from functions_luca import *
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

LR = 0.01
MAX_EPOCHS = 10000
SIGMA = 0.5  # 0.5
SINE = True
NOISE = False
SIGMA_NOISE = 0.1
BATCH = False
ES = True
PATIENCE = 1
PLOT = False
# MLP params
MLP_ = False

# competitive learning constants - only if BATCH = False
MAX_EPOCHS_CL = 10
COMPETITIVE = True
# strategy to avoid dead units
MORE_THAN_ONE_WINNER = False
# NUM_OF_WINNERS = int(NUM_NODES / 4)
# learning rate for competitive learning part
LR_CL = 0.01

BALLISTIC_DATA = True
# percentage of val set -> only if BALLISTIC_DATA = True
val_p = 0.2


def main(NUM_NODES):
    if COMPETITIVE:
        print("CL")
        if MORE_THAN_ONE_WINNER:
            print("more than one winner")
    else:
        print("NOT CL")

    NUM_OF_WINNERS = int(NUM_NODES / 2)
    if BALLISTIC_DATA:
        # data -> ballistic experiments
        train_data = np.genfromtxt('data/ballist.dat')
        test_data = np.genfromtxt('data/balltest.dat')

        col_of_separation_patterns_targets = 2

        patterns, targets, val_patterns, val_targets = train_val_split(train_data, val_p,
                                                                       col_of_separation_patterns_targets)
        test_patterns = test_data[:, :col_of_separation_patterns_targets]
        test_targets = test_data[:, col_of_separation_patterns_targets:]

        mu = init_means(NUM_NODES, BALLISTIC_DATA, patterns)
        w = init_weights(NUM_NODES, BALLISTIC_DATA)
    else:
        # data -> sine or square functions
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
        init_mu = np.copy(mu)
        w = init_weights(NUM_NODES)

    phi_mat = np.zeros((NUM_NODES, patterns.shape[0]))

    for i in range(NUM_NODES):
        for j in range(patterns.shape[0]):
            phi_mat[i][j] = phi(euclidean_distance(mu[i], patterns[j]), SIGMA)

    if MLP_:
        v_MLP, w_MLP, preds = MLP(np.transpose(patterns), targets.reshape(targets.shape[0]),
                                  np.transpose(val_patterns), val_targets.reshape(val_targets.shape[0]),
                                  MAX_EPOCHS, NUM_NODES, LR, ES, PATIENCE)
        if PLOT:
            plot(patterns, targets, np.transpose(preds), LR, NUM_NODES, MAX_EPOCHS,
                 MLP=True, es=ES, patience=PATIENCE)

    elif BATCH:
        w = train_batch(phi_mat, targets)
    else:
        # sequential - on-line
        if COMPETITIVE:
            # competitive learning
            competitive_learning(patterns, mu, LR_CL, NUM_NODES, MORE_THAN_ONE_WINNER, NUM_OF_WINNERS, MAX_EPOCHS_CL)

        w, train_errors, val_errors, epoch = train_seq(patterns, targets, w, MAX_EPOCHS, SIGMA, mu, LR, PLOT, ES,
                                                       val_patterns,
                                                       val_targets, PATIENCE, BALLISTIC_DATA)
        if PLOT:
            # plot learning curves
            plot_train_val(train_errors, val_errors, ballistic_data=BALLISTIC_DATA, lr=LR, num_nodes=NUM_NODES,
                           max_epochs=MAX_EPOCHS,
                           batch=False, cl=COMPETITIVE, es=ES, patience=PATIENCE, MLP=False,
                           lr_cl=LR_CL, epochs_cl=MAX_EPOCHS_CL, more_winners=MORE_THAN_ONE_WINNER,
                           import_data=BALLISTIC_DATA)

    if BALLISTIC_DATA:
        preds = get_continuous_predictions(mu, w, SIGMA, patterns)
        test_preds = get_continuous_predictions(mu, w, SIGMA, test_patterns)
        val_preds = get_continuous_predictions(mu, w, SIGMA, val_patterns)
    else:
        # sine or square function
        if SINE:
            # sine function
            preds = get_continuous_predictions(mu, w, SIGMA, patterns)
            test_preds = get_continuous_predictions(mu, w, SIGMA, test_patterns)
            val_preds = get_continuous_predictions(mu, w, SIGMA, val_patterns)
        else:
            # square function
            preds = get_discrete_predictions(mu, w, SIGMA, patterns)
            test_preds = get_discrete_predictions(mu, w, SIGMA, test_patterns)
            val_preds = get_discrete_predictions(mu, w, SIGMA, val_patterns)

    resid_error_test_set = residual_error(test_preds, test_targets)
    mse_test_set = mse(test_preds, test_targets)

    # print("Residual test set: {}".format(resid_error_test_set))
    # print("MSE test set: {}".format(mse_test_set))

    if PLOT:
        # training set
        if COMPETITIVE:
            comp_param = True
        else:
            comp_param = False
        if BALLISTIC_DATA:
            init_mu = None
        plot(patterns, targets, preds, LR, NUM_NODES, MAX_EPOCHS,
             batch=BATCH, cl=COMPETITIVE, lr_cl=LR_CL, es=ES, patience=PATIENCE, epochs_cl=MAX_EPOCHS_CL,
             more_winners=MORE_THAN_ONE_WINNER, import_data=BALLISTIC_DATA, centroids=mu, validation=False, test=False,
             centroids_init=init_mu, competitive=comp_param)

        # test set
        plot(test_patterns, test_targets, test_preds, LR, NUM_NODES, MAX_EPOCHS,
             batch=BATCH, cl=COMPETITIVE, lr_cl=LR_CL, es=ES, patience=PATIENCE, epochs_cl=MAX_EPOCHS_CL,
             more_winners=MORE_THAN_ONE_WINNER, import_data=BALLISTIC_DATA, centroids=mu, test=True)
        '''
        # # validation set
        plot(val_patterns, val_targets, val_preds, LR, NUM_NODES, MAX_EPOCHS,
             batch=BATCH, cl=COMPETITIVE, lr_cl=LR_CL, es=ES, patience=PATIENCE, epochs_cl=MAX_EPOCHS_CL,
             more_winners=MORE_THAN_ONE_WINNER, import_data=BALLISTIC_DATA, centroids=mu, validation=True)
        '''
    return resid_error_test_set, mse_test_set


if __name__ == '__main__':

    #np.random.seed(2)

    NUM_NODES_LIST = [24]#, 4, 6, 8, 13, 20]
    NUM_TRIALS = 10

    for NUM_NODES in NUM_NODES_LIST:
        # average error on 10 iterations
        res_error = 0
        mse_error = 0
        for i in range(NUM_TRIALS):
            (res_error_upd, mse_err_upd) = main(NUM_NODES)
            res_error += res_error_upd
            mse_error += mse_err_upd

        print(NUM_NODES, res_error / NUM_TRIALS, mse_error / NUM_TRIALS)
