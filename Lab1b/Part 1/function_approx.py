import numpy as np
import matplotlib.pyplot as plt
from fontTools.misc.plistlib import _array_element

n_samples = 100


def fun(x, y):
    return np.exp(-(x**2 + y**2)*0.1) - 0.5


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


def main():

    patterns, targets = generate_2d_gaussian()
    plot_3d(patterns, targets)


if __name__ == '__main__':
    main()
    