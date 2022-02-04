import numpy as np
import matplotlib.pyplot as plt

from numpy.random import normal
from main import forward_pass, backward_pass, weight_update, MSE

HIDDEN_NODES = 3
EPOCHS = 1000
LEARNING_RATE = 0.01

def fun(x, y):
    return np.exp(- (x ** 2 + y ** 2) * 0.1) - 0.5


def generate_2d_gaussian(from_xy=-0.5, to_xy=0.5, n_samples=n_samples):
    x = np.linspace(from_xy, to_xy, n_samples)
    y = np.linspace(from_xy, to_xy, n_samples)
    targets = np.array([[fun(x_elem, y_elem) for x_elem in x] for y_elem in y])
    targets = targets.reshape((n_samples**2, 1))
    [xx, yy] = np.meshgrid(x, y)
    patterns = np.transpose(np.concatenate((xx.reshape(1, n_samples**2), yy.reshape(1, n_samples**2))))#np.array([(xx[i], yy[i]) for i in range(len(xx))])

    return patterns, targets


def plot_3d(patterns, targets):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
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

def save_errors(o_out, targets, MSE_errors,):
    MSE_errors.append(MSE(o_out, targets))
    

def main():
    patterns, targets = generate_2d_gaussian()

    w = normal(0, 1, [HIDDEN_NODES, 2])
    v = normal(0, 1, HIDDEN_NODES).reshape(1, HIDDEN_NODES)

    MSE_errors = []
    for i_epoch in range(EPOCHS):
        h_in, h_out, o_in, o_out = forward_pass(patterns, w, v)
        save_errors(o_out, targets, MSE_errors)

        print(f"EPOCH {i_epoch:4d} | training_mse = {MSE(o_out, targets):4.2f} |")

        delta_h, delta_o = backward_pass(v, targets, h_in, o_out, o_in)
        w, dw = weight_update(w, patterns, delta_h, lr=LEARNING_RATE, momentum=False, d_old=dw)
        v, dv = weight_update(v, h_out, delta_o, lr=LEARNING_RATE, momentum=False, d_old=dv)



if __name__ == '__main__':
    main()
