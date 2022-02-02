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
n_epochs = 200



def f(x):
    return (2 / (1 + np.exp(-x))) - 1


def f_prime(x):
    return ((1 + f(x)) * (1 - f(x))) * 0.5


def get_patterns():
    # create class A (disjoint) and B, with specified global means and cov (diagonal)
    # return classes with bias coordinate
    classA_1 = multivariate_normal(m_A, [[sigma_A, 0], [0, sigma_A]], int(n * 0.5))
    classA_2 = multivariate_normal([-m_A[0], -m_A[1]], [[sigma_A, 0], [0, sigma_A]], int(n * 0.5))

    classA = np.concatenate((classA_1,classA_2))
    classB = multivariate_normal(m_B, [[sigma_B,0],[0, sigma_B]], n)

    patterns = np.array([[x[0], x[1], bias] for x in classA] + [[x[0], x[1], bias] for x in classB])
    targets = np.array([1 for x in classA] + [-1 for x in classB])

    return patterns.transpose(), targets

def forward_pass(patterns, w, v):
    h_in = w @ patterns
    h_out = f(h_in)
    o_in = v @ h_out
    o_out = f(o_in)
    return h_in, h_out, o_in, o_out

def save_errors(o_out,targets, MSE_errors, miscl_error):
    print(MSE(o_out, targets))
    miscl_error.append(misclass_rate(o_out, targets))
    MSE_errors.append(MSE(o_out, targets))

def backward_pass(v, targets, h_in, o_out, o_in):

    delta_o = np.multiply(np.subtract(o_out, targets), f_prime(o_in))
    v = v.reshape(1, hidden_nodes)
    delta_o = delta_o.reshape(1, delta_o.shape[1])
    delta_h = np.multiply((v.transpose() @ delta_o), f_prime(h_in))

    return delta_h, delta_o

def weight_update(weights, inputs, delta, lr, momentum=False, alpha=0.9, d_old=None):
    if momentum:
        print("momentum")
        d = (d_old * alpha) - (delta * np.transpose(inputs)) * (1 - alpha)
    else:
        d = delta @ inputs.transpose()
    weights += np.multiply(d, lr)
    return weights, d

def MSE(preds, targets):
    errors = preds - targets
    return np.sum(errors ** 2) / len(preds)

def misclass_rate(o_out, targets):
    error_rate = 0
    preds = np.where(o_out > 0, 1, -1)[0]
    for i in range(len(preds)):
        if preds[i] != targets[i]:
            error_rate += 1
    return error_rate/len(preds)

def plot_errors(MSE_errors, miscl_errors):
    plt.plot(np.arange(len(MSE_errors)), MSE_errors)
    plt.show()

def main():
    patterns, targets = get_patterns()

    w = normal(0, 1, [hidden_nodes, 3])
    v = normal(0, 1, hidden_nodes).reshape(1, 3)

    dw = 0
    dv = 0

    MSE_errors = []
    miscl_errors = []

    for i_epoch in range(n_epochs):
        print("------ EPOCH {} ------".format(i_epoch))

        h_in, h_out, o_in, o_out = forward_pass(patterns, w, v)
        save_errors(o_out, targets,MSE_errors, miscl_errors)
        delta_h, delta_o = backward_pass(v, targets, h_in, o_out, o_in)
        w, dw = weight_update(w, patterns, delta_h, lr=learning_rate, momentum=False, d_old=dw)
        v, dv = weight_update(v, h_out, delta_o, lr=learning_rate, momentum=False, d_old=dv)

    plot_errors(MSE_errors, miscl_errors)

if __name__ == '__main__':
    main()
