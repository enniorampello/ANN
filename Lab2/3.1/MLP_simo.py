import numpy as np
from numpy.random import multivariate_normal, normal
import matplotlib.pyplot as plt
from functions import *


def f(x):
    return (2 / (1 + np.exp(-x))) - 1

def f_prime(x):
    return ((1 + f(x)) * (1 - f(x))) * 0.5

def forward_pass(patterns, w, v):
    h_in = w @ patterns
    h_out = f(h_in)
    o_in = v @ h_out
    o_out = f(o_in)
    return h_in, h_out, o_in, o_out

def backward_pass(v, targets, h_in, o_out, o_in, hidden_nodes):
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
    weights -= np.multiply(d, lr)
    return weights, d

def MSE(preds, targets):
    errors = preds - targets
    return np.sum(errors ** 2) / len(preds)

def MLP(patterns, targets, val_patterns, val_targets, max_epochs, hidden_nodes, learning_rate, val):
    bias = np.ones(patterns.shape)
    patterns = np.concatenate((patterns, bias))

    w = normal(0, 1, [hidden_nodes, patterns.shape[0]])
    v = normal(0, 1, hidden_nodes).reshape(1, hidden_nodes)

    dw = 0
    dv = 0

    train_errors = []
    val_errors = []

    for i_epoch in range(max_epochs):
            h_in, h_out, o_in, o_out = forward_pass(patterns, w, v)

            e = np.sum(np.abs(o_out-targets))/len(targets)
            e = MSE(o_out, targets)
            train_errors.append(e)

            if val:
                _, _, _, o_out_val = forward_pass(val_patterns, w, v)
                val_errors.append(sum(abs(o_out_val - val_targets)) / len(val_targets))
            print(f"EPOCH {i_epoch:4d} | training_mse = {e:.2f} |")
            delta_h, delta_o = backward_pass(v, targets, h_in, o_out, o_in, hidden_nodes)

            w, dw = weight_update(w, patterns, delta_h, lr=learning_rate, momentum=False, d_old=dw)
            v, dv = weight_update(v, h_out, delta_o, lr=learning_rate, momentum=False, d_old=dv)

    _, _, _, preds = forward_pass(patterns, w, v)
    return v, w, preds

