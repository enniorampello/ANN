import numpy as np
import matplotlib.pyplot as plt

from numpy.random import normal
from main import forward_pass, backward_pass, weight_update, MSE

HIDDEN_NODES = 25
EPOCHS = 1000
LEARNING_RATE = 0.001
N_SAMPLES = 0
BIAS = 1
BATCH_SIZE = 32

def fun(x, y):
    return np.exp(- (x ** 2 + y ** 2) * 0.1) - 0.5


def generate_2d_gaussian(from_xy=-5, to_xy=5.01, step=0.5, n_samples=N_SAMPLES):
    x = np.arange(from_xy, to_xy, step)
    y = np.arange(from_xy, to_xy, step)
    global N_SAMPLES
    N_SAMPLES = len(x)
    targets = np.array([[fun(x_elem, y_elem) for x_elem in x] for y_elem in y])
    targets = targets.reshape((N_SAMPLES ** 2,))
    [xx, yy] = np.meshgrid(x, y)
    patterns = np.transpose(np.concatenate((xx.reshape(1, N_SAMPLES ** 2), yy.reshape(1, N_SAMPLES ** 2))))
    patterns = patterns.transpose()
    patterns = np.vstack([patterns, [BIAS for _ in range(patterns.shape[1])]])

    return patterns, targets


def plot_3d(patterns, targets):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-0.7, 0.7)
    patterns_t = np.transpose(patterns)
    X, Y = patterns_t[0], patterns_t[1]
    X = X.reshape((N_SAMPLES, N_SAMPLES))
    Y = Y.reshape((N_SAMPLES, N_SAMPLES))

    zs = targets
    Z = zs.reshape(X.shape)
    ax.plot_surface(X, Y, Z)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()


def save_errors(o_out, targets, MSE_errors, ):
    MSE_errors.append(MSE(o_out, targets))


def main():
    patterns, targets = generate_2d_gaussian()

    w = normal(0, 1, [HIDDEN_NODES, 3])
    v = normal(0, 1, HIDDEN_NODES).reshape(1, HIDDEN_NODES)

    dw = 0
    dv = 0
    plot_3d(patterns.transpose(), targets)
    MSE_errors = []



    for i_epoch in range(EPOCHS):
        for i_batch in range(int(patterns.shape[1]/BATCH_SIZE)):
            idx_start = i_batch * BATCH_SIZE
            if i_batch * BATCH_SIZE + BATCH_SIZE > patterns.shape[1]:
                idx_end = patterns.shape[1]
            else:
                idx_end = i_batch * BATCH_SIZE + BATCH_SIZE
            h_in, h_out, o_in, o_out = forward_pass(patterns[:, idx_start:idx_end],  w, v)
            # save_errors(o_out, targets, MSE_errors)


            # print(f"EPOCH {i_epoch:4d} | training_mse = {MSE(o_out, targets[idx_start:idx_end]):4.2f} |")

            delta_h, delta_o = backward_pass(v, targets[idx_start:idx_end], h_in, o_out, o_in, HIDDEN_NODES)
            w, dw = weight_update(w, patterns[:, idx_start:idx_end], delta_h, lr=LEARNING_RATE, momentum=False, d_old=dw)
            v, dv = weight_update(v, h_out, delta_o, lr=LEARNING_RATE, momentum=False, d_old=dv)
        h_in, h_out, o_in, o_out = forward_pass(patterns, w, v)
        print(f"EPOCH {i_epoch:4d} | training_mse = {MSE(o_out, targets):4.2f} |")


    # 3d-plot
    plot_3d(patterns.transpose(), o_out)


if __name__ == '__main__':
    main()
