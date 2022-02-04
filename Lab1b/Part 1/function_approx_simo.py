import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set()

from numpy.random import normal
from main import forward_pass, backward_pass, weight_update, MSE

HIDDEN_NODES = 4
EPOCHS = 5000
LEARNING_RATE = 0.0001
STEP = 0.5

BIAS = -1


def fun(x, y):
    return np.exp(- (x ** 2 + y ** 2) * 0.1) - 0.5


def generate_2d_gaussian(from_xy=-5, to_xy=5.01):
    x = np.arange(from_xy, to_xy, STEP)
    y = np.arange(from_xy, to_xy, STEP)
    n_samples = len(x)
    targets = np.array([[fun(x_elem, y_elem) for x_elem in x] for y_elem in y])
    targets = targets.reshape((n_samples ** 2,))
    [xx, yy] = np.meshgrid(x, y)
    patterns = np.transpose(np.concatenate((xx.reshape(1, n_samples ** 2), yy.reshape(1, n_samples ** 2))))

    return patterns, targets, n_samples


def plot_3d(patterns, targets, n_samples):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-0.7, 0.7)
    patterns_t = np.transpose(patterns)
    X, Y = patterns_t[0], patterns_t[1]
    X = X.reshape((n_samples, n_samples))
    Y = Y.reshape((n_samples, n_samples))

    zs = targets
    Z = zs.reshape(X.shape)
    ax.plot_surface(X, Y, Z)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()


def save_errors(o_out, targets, MSE_errors):
    MSE_errors.append(MSE(o_out, targets))


def main():
    patterns, targets, n_samples = generate_2d_gaussian()
    patterns = patterns.transpose()
    patterns = np.vstack([patterns, [BIAS for _ in range(patterns.shape[1])]])
    print(patterns.shape)
    w = normal(0, 1, [HIDDEN_NODES, 3])
    v = normal(0, 1, HIDDEN_NODES).reshape(1, HIDDEN_NODES)


    dw = 0
    dv = 0

    MSE_errors = []
    for i_epoch in range(EPOCHS):
        h_in, h_out, o_in, o_out = forward_pass(patterns, w, v)
        save_errors(o_out, targets, MSE_errors)

        if i_epoch == EPOCHS - 1:
        # 3d-plot
            plot_3d(patterns.transpose(), o_out)

        print(f"EPOCH {i_epoch:4d} | training_mse = {MSE(o_out, targets):4.2f} |")

        delta_h, delta_o = backward_pass(v, targets, h_in, o_out, o_in)
        w, dw = weight_update(w, patterns, delta_h, lr=LEARNING_RATE, momentum=False, d_old=dw)
        v, dv = weight_update(v, h_out, delta_o, lr=LEARNING_RATE, momentum=False, d_old=dv)


if __name__ == '__main__':
    main()