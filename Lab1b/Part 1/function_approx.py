import numpy as np
import matplotlib.pyplot as plt

from numpy.random import normal
from main import forward_pass, backward_pass, weight_update, MSE

HIDDEN_NODES = 120
EPOCHS = 3000
LEARNING_RATE = 0.001
N_SAMPLES = 0
BIAS = 1

# plot constants
X_MIN = -5
X_MAX = -X_MIN
Y_MIN = X_MIN
Y_MAX = X_MAX
Z_MIN = -0.7
Z_MAX = -Z_MIN


def bell_gaussian_func(x, y):
    return np.exp(- (x ** 2 + y ** 2) * 0.1) - 0.5


def generate_2d_gaussian(from_xy=-5, to_xy=5.01, step=0.5):
    x = np.arange(from_xy, to_xy, step)
    y = np.arange(from_xy, to_xy, step)
    global N_SAMPLES
    N_SAMPLES = len(x)
    targets = np.array([[bell_gaussian_func(x_elem, y_elem) for x_elem in x] for y_elem in y])
    targets = targets.reshape((N_SAMPLES ** 2,))
    [xx, yy] = np.meshgrid(x, y)
    patterns = np.transpose(np.concatenate((xx.reshape(1, N_SAMPLES ** 2), yy.reshape(1, N_SAMPLES ** 2))))
    patterns = patterns.transpose()
    patterns = np.vstack([patterns, [BIAS for _ in range(patterns.shape[1])]])

    return patterns, targets


def plot_3d(patterns, targets, i_epoch):
    fig = plt.figure()
    plt.title("network approximation of function - epoch {}".format(i_epoch + 1))
    plt.axis('off')
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_zlim(Z_MIN, Z_MAX)
    patterns_t = np.transpose(patterns)
    X, Y = patterns_t[0], patterns_t[1]
    X = X.reshape((N_SAMPLES, N_SAMPLES))
    Y = Y.reshape((N_SAMPLES, N_SAMPLES))

    zs = targets
    Z = zs.reshape(X.shape)
    ax.plot_surface(X, Y, Z)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


def save_errors(o_out, targets, MSE_errors,):
    MSE_errors.append(MSE(o_out, targets))
    

def main():
    patterns, targets = generate_2d_gaussian()

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
            plot_3d(patterns.transpose(), o_out, i_epoch)

        print(f"EPOCH {i_epoch:4d} | training_mse = {MSE(o_out, targets):4.2f} |")

        delta_h, delta_o = backward_pass(v, targets, h_in, o_out, o_in, HIDDEN_NODES)
        w, dw = weight_update(w, patterns, delta_h, lr=LEARNING_RATE, momentum=False, d_old=dw)
        v, dv = weight_update(v, h_out, delta_o, lr=LEARNING_RATE, momentum=False, d_old=dv)


if __name__ == '__main__':
    main()
