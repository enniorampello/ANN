import numpy as np
import matplotlib.pyplot as plt

def sin(patterns, start=0, stop=2*np.pi):
    """

    Args:
        start (int, optional): starting point of the sequence. Set to 0 for training set and 0.05 for val and np.pi+0.5 for test. Defaults to 0.
        stop ([type], optional): ending point of the sequence. Only use with test/val. If val, set to np.pi, else 2*np.pi. Defaults to 2*np.pi.

    Returns:
        np.array: array containing the sequence of points
    """
    return np.sin(2*patterns).reshape(len(patterns), 1)

def square(patterns, start=0, stop=2*np.pi):
    return np.array([1 if np.sin(2*x) >= 0 else -1 for x in patterns]).reshape(int(stop/0.1), 1)

def phi(r, sigma):
    return np.exp(-(r**2)/(2*sigma**2))

def init_means(num_nodes):
    return np.linspace(0, 2*np.pi, num_nodes).reshape(num_nodes, 1)

def init_weights(num_nodes):
    return np.random.normal(0, scale=1, size=(num_nodes, 1))

def add_noise(points, sigma):
    noise = np.random.normal(0, scale=sigma, size=(len(points), 1))
    return points + noise

def train_batch(phi_mat, targets):
    return np.linalg.inv(phi_mat @ phi_mat.T) @ phi_mat @ targets

def train_seq(patterns, targets, w, max_epochs, sigma, mu, lr, plot, ES, val_patterns, val_targets, patience):

    if plot:
        pred = [forward_pass(x, mu, w, sigma)[1] for x in patterns]
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        line, = ax.plot(patterns, pred)
        fig.canvas.draw()
        plt.show(block=False)

    val_errors = []
    patience_counter = 0
    for epoch in range(max_epochs):
        error = 0
        for pattern, target in zip(patterns, targets):
            h_out, _ = forward_pass(pattern, mu, w, sigma)
            w = update_weights(target, h_out, w, lr)


            error += abs(target - np.sum(h_out * w))
        error /= patterns.shape[0]


        if ES:
            val_error = 0
            for pattern, target in zip(val_patterns, val_targets):
                h_out, val_pred = forward_pass(pattern, mu, w, sigma)
                val_error += abs(target - np.sum(h_out * w))
            val_error /= val_patterns.shape[0]
            val_errors.append(val_error)
            if epoch > 1 and val_errors[-1][0] > val_errors[-2][0]:
                patience_counter += 1
            else:
                patience_counter = 0
            if patience_counter >= patience:
                print('Early stopping!!')
                return w

        print(f'EPOCH {epoch}\t| error {error[0]} | val error {val_errors[-1][0]}')

        if epoch % 10 == 0 and plot:
            #clear_output(wait=False)
            pred = [forward_pass(x, mu, w, sigma)[1] for x in patterns]
            #fig = plt.figure()
            line.set_xdata(patterns)
            line.set_ydata(pred)
            ax.relim()
            ax.autoscale_view(True,True,True)
            fig.canvas.draw()
            plt.pause(0.01)
    return w

def forward_pass(pattern, mu, w, sigma):
    h_in = np.abs(mu - pattern)
    h_out = phi(h_in, sigma)
    o_out = np.sum(w * h_out)
    return h_out, o_out

def update_weights(target, h_out, w, lr):
    w += lr * (target - np.sum(h_out * w)) * h_out
    return w

def print_function(f, start=0, stop=2*np.pi):
    x = np.linspace(start, start+stop, int(stop/0.1))
    plt.figure()
    plt.plot(x, f)
    plt.show()