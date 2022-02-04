import numpy as np

from numpy.random import normal
from main import forward_pass, backward_pass, weight_update, MSE

HIDDEN_NODES = 3
EPOCHS = 1000
LEARNING_RATE = 0.01

def generate_2d_gaussian(from_xy=-0.5, to_xy=0.5, n_samples=100):
    x = np.linspace(from_xy, to_xy, n_samples)
    y = np.linspace(from_xy, to_xy, n_samples)
    
    targets = np.subtract(np.exp(-(np.add(np.square(x), np.square(y))/10)), 0.5)
    patterns = np.array([[]] for i in range(len(x))])

    return patterns, targets

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