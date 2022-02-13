
from functions import *
from numpy.random import normal

LR = 0.001
NUM_NODES = 12
MAX_EPOCHS = 10000
SINE = True

np.random.seed(2)

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

def main():
    patterns = np.linspace(0, 2 * np.pi, int(2 * np.pi / 0.1)).reshape(int(2 * np.pi / 0.1), 1)
    patterns = np.array([(pattern, 1) for pattern in patterns])
    if SINE:
        targets = sin(patterns)
    else:
        targets = square(patterns)
    patterns = patterns.T
    targets = targets.T
    
    w = normal(0, 1, [NUM_NODES, 2])
    v = normal(0, 1, NUM_NODES).reshape(1, NUM_NODES)

    dw = 0
    dv = 0
    for i_epoch in range(MAX_EPOCHS):
        h_in, h_out, o_in, o_out = forward_pass(patterns, w, v)

        print(f"EPOCH {i_epoch:4d} | training_mse = {MSE(o_out, targets):4.2f} |")

        delta_h, delta_o = backward_pass(v, targets, h_in, o_out, o_in, NUM_NODES)
        w, dw = weight_update(w, patterns, delta_h, lr=LR, momentum=False,
                            d_old=dw)
        v, dv = weight_update(v, h_out, delta_o, lr=LR, momentum=False, d_old=dv)
    h_in, h_out, o_in, o_out = forward_pass(patterns, w, v)
    # save_errors(o_out, targets, MSE_errors)
# 
    # if val:
    #     _, _, _, o_out_val = forward_pass(val_patterns, w, v)
    print(o_out.shape, patterns.shape)
    print_function(o_out.reshape(62,))

if __name__ == '__main__':
    main()