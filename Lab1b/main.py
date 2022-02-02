import numpy as np
from numpy.random import multivariate_normal, normal
import matplotlib.pyplot as plt

n = 100
m_A = [1.0, 0.3]
m_B = [0.0, -0.1]

sigma_A = 0.2
sigma_B = 0.3

bias = 1
hidden_nodes = 3
learning_rate = 0.001
n_epochs = 20

def f(x):
    return (2 / (1 + np.exp(-x))) - 1


def f_prime(x):
    return ((1 + f(x)) * (1 - f(x))) / 2


def get_patterns():
    # create class A (disjoint) and B, with specified global means and cov (diagonal)
    # return classes with bias coordinate
    classA_1 = multivariate_normal(m_A, [[sigma_A,0],[0, sigma_A]], int(n * 0.5))
    classA_2 = multivariate_normal([-m_A[0],-m_A[1]], [[sigma_A,0],[0, sigma_A]], int(n * 0.5))

    classA = np.concatenate((classA_1,classA_2))
    classB = multivariate_normal(m_B, [[sigma_B,0],[0, sigma_B]], n)

    patterns = np.array([[x[0], x[1], bias] for x in classA] + [[x[0], x[1], bias] for x in classB])
    targets = np.array([1 for x in classA] + [-1 for x in classB])

    return patterns.transpose(), targets


def weight_update(weights, inputs, delta, lr, momentum=False, alpha=0.9, d_old=None):
    if momentum:
        print("momentum")
        d = (d_old * alpha) - (delta * np.transpose(inputs)) * (1 - alpha)
    else:
        d = delta * np.transpose(inputs)

    weights += (d * lr)
    return weights, d

def backward_pass(V, targets, h_in, out_out, out_in):
    delta_o = np.multiply(np.subtract(out_out, targets), f_prime(out_in))
    delta_h = np.multiply((V @ delta_o), h_in)

    return delta_h, delta_o


def MSE(preds, targets):
    errors = preds - targets
    return sum(errors ** 2) / len(preds)


def main():
    patterns, targets = get_patterns()
    W = normal(0, 1, [hidden_nodes, 3])
    V = normal(0, 1, hidden_nodes)
    for i_epoch in range(n_epochs):

        H = f(np.dot(W, patterns))
        O = f(np.dot(V, H))


if __name__ == '__main__':
    main()
