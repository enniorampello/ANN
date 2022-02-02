import numpy as np
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal

n_data = 100
m_a = [1.0, 0.3]
m_b = [0.0, -0.1]
sigma_a = 0.2
sigma_b = 0.3
n_nodes_hidden = 4
learning_rate = 0.001
n_epochs = 20


def f(x):
    return (2 / (1 + np.exp(-x))) - 1


def f_prime(x):
    return ((1 + f(x)) * (1 - f(x))) / 2


def weight_update(weights, inputs, delta, lr, momentum=False, alpha=0.9, d_old=None):
    if momentum:
        print("momentum")
        d = (d_old * alpha) - (delta * np.transpose(inputs)) * (1 - alpha)
    else:
        d = delta * np.transpose(inputs)

    weights += (d * learning_rate)
    return weights

def main():
    classA_1 = multivariate_normal(m_a, [[sigma_a, 0], [0, sigma_a]], int(n_data * 0.5))
    classA_2 = multivariate_normal([-m_a[0], -m_a[1]], [[sigma_a, 0], [0, sigma_a]], int(n_data * 0.5))

    classA = np.concatenate((classA_1, classA_2))
    classB = multivariate_normal(m_b, [[sigma_b, 0], [0, sigma_b]], n_data)

    plt.scatter(classA[:, 0], classA[:, 1])
    plt.scatter(classB[:, 0], classB[:, 1])

    patterns = np.array([[[x[0], x[1], 1] for x in classA] + [[x[0], x[1], -1] for x in classB]])
    plt.show()


if __name__ == '__main__':
    main()
