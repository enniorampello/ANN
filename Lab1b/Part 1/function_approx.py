import numpy as np
import matplotlib.pyplot as plt

from numpy.random import normal
from main import forward_pass, backward_pass, weight_update, MSE

HIDDEN_NODES = 25
EPOCHS = 1000
LEARNING_RATE = 0.001
STEP = 0.5
BIAS = 1
BATCH_SIZE = 32

val = True
val_p = 0.2

# plot constants
X_MIN = -5
X_MAX = -X_MIN
Y_MIN = X_MIN
Y_MAX = X_MAX
Z_MIN = -0.7
Z_MAX = -Z_MIN


def bell_gaussian_func(x, y):
    return np.exp(- (x ** 2 + y ** 2) * 0.1) - 0.5


def generate_2d_gaussian(from_xy=-5, to_xy=5.01):
    x = np.arange(from_xy, to_xy, STEP)
    y = np.arange(from_xy, to_xy, STEP)
    n_samples = len(x)

    targets = np.array([[bell_gaussian_func(x_elem, y_elem) for x_elem in x] for y_elem in y])
    targets = targets.reshape((n_samples ** 2,))

    [xx, yy] = np.meshgrid(x, y)
    patterns = np.concatenate((xx.reshape(1, n_samples ** 2), yy.reshape(1, n_samples ** 2), BIAS * np.ones((1, n_samples ** 2))))

    return patterns, targets, n_samples


def plot_3d(patterns, targets, n_samples, i_epoch):
    fig = plt.figure()
    plt.title("network approximation of function - epoch {}".format(i_epoch + 1))
    plt.axis('off')
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_zlim(Z_MIN, Z_MAX)
    patterns_t = np.transpose(patterns)
    X, Y = patterns_t[0], patterns_t[1]
    X = X.reshape((n_samples, n_samples))
    Y = Y.reshape((n_samples, n_samples))

    zs = targets
    Z = zs.reshape(X.shape)
    ax.plot_surface(X, Y, Z)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


def train_val_split(patterns, targets, val_perc):
    n = patterns.shape[1]

    merged = np.vstack([patterns, targets.transpose()]).transpose()
    np.random.shuffle(merged)

    val = merged[:int(n * val_perc)]
    train = merged[int(n * val_perc):]

    train_patterns = train[:, :-1].transpose()
    train_labels = train[:, -1]

    val_patterns = val[:, :-1].transpose()
    val_labels = val[:, -1]

    return train_patterns, train_labels, val_patterns, val_labels


def save_errors(o_out, targets, mse_errors,):
    mse_errors.append(MSE(o_out, targets))


def plot_mse_error(mse_errors, i_epoch):
    plt.title('Learning curve - function approximation - epoch {}'.format(i_epoch))
    plt.ylabel('MSE')
    plt.xlabel('epochs')
    plt.plot(mse_errors, color='red', label='MSE')
    plt.show()
    

def main():
    patterns, targets, n_samples = generate_2d_gaussian()

    if val:
        train_patterns, train_labels, val_patterns, val_labels = train_val_split(patterns, targets, val_p)

    w = normal(0, 1, [HIDDEN_NODES, 3])
    v = normal(0, 1, HIDDEN_NODES).reshape(1, HIDDEN_NODES)

    dw = 0
    dv = 0
    
    MSE_errors = []
    for i_epoch in range(EPOCHS):
        for i_batch in range(int(patterns.shape[1] / BATCH_SIZE)):
            idx_start = i_batch * BATCH_SIZE
            if i_batch * BATCH_SIZE + BATCH_SIZE > patterns.shape[1]:
                idx_end = patterns.shape[1]
            else:
                idx_end = i_batch * BATCH_SIZE + BATCH_SIZE
            h_in, h_out, o_in, o_out = forward_pass(patterns[:, idx_start:idx_end], w, v)


            # print(f"EPOCH {i_epoch:4d} | training_mse = {MSE(o_out, targets[idx_start:idx_end]):4.2f} |")

            delta_h, delta_o = backward_pass(v, targets[idx_start:idx_end], h_in, o_out, o_in, HIDDEN_NODES)
            w, dw = weight_update(w, patterns[:, idx_start:idx_end], delta_h, lr=LEARNING_RATE, momentum=False,
                                  d_old=dw)
            v, dv = weight_update(v, h_out, delta_o, lr=LEARNING_RATE, momentum=False, d_old=dv)
        h_in, h_out, o_in, o_out = forward_pass(patterns, w, v)
        save_errors(o_out, targets, MSE_errors)
        print(f"EPOCH {i_epoch:4d} | training_mse = {MSE(o_out, targets):4.2f} |")

    # plot approximation of the function
    plot_3d(patterns.transpose(), o_out, n_samples, i_epoch)
    # plot learning curve
    plot_mse_error(MSE_errors, i_epoch)


if __name__ == '__main__':
    main()
