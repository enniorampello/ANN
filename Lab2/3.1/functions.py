import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def sin(patterns, start=0, stop=2 * np.pi):
    """

    Args:
        start (int, optional): starting point of the sequence. Set to 0 for training set and 0.05 for val and np.pi+0.5 for test. Defaults to 0.
        stop ([type], optional): ending point of the sequence. Only use with test/val. If val, set to np.pi, else 2*np.pi. Defaults to 2*np.pi.

    Returns:
        np.array: array containing the sequence of points
    """
    return np.sin(2 * patterns).reshape(len(patterns), 1)


def sin_prime(patterns):
    return 2 * np.cos(2 * patterns).reshape(len(patterns), 1)


def square(patterns, start=0, stop=2 * np.pi):
    return np.array([1 if np.sin(2 * x) >= 0 else -1 for x in patterns]).reshape(len(patterns), 1)


def phi(r, sigma):
    return np.exp(-(r ** 2) / (2 * sigma ** 2))


def init_means(num_nodes, import_data=False, patterns=None):
    if import_data:
        kmeans = KMeans(n_clusters=num_nodes, random_state=0).fit(patterns)
        return kmeans.cluster_centers_
    else:
        return np.linspace(0, 2*np.pi, num_nodes).reshape(num_nodes, 1)


def init_weights(num_nodes, import_data=False):
    if import_data:
        return np.random.normal(0, scale=1, size=(num_nodes, 2))
    else:
        return np.random.normal(0, scale=1, size=(num_nodes, 1))


def add_noise(points, sigma):
    noise = np.random.normal(0, scale=sigma, size=(len(points), 1))
    return points + noise


def train_batch(phi_mat, targets):
    return np.linalg.inv(phi_mat @ phi_mat.T) @ phi_mat @ targets


def train_seq(patterns, targets, w, max_epochs, sigma, mu, lr, plot, ES,
              val_patterns, val_targets, patience, import_data):

    if plot and not import_data:
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
            h_out, x = forward_pass(pattern, mu, w, sigma)

            w = update_weights(target, h_out, w, lr)
            error += abs(target - np.sum(h_out * w))
        error /= patterns.shape[0]

        val_error = 0
        for pattern, target in zip(val_patterns, val_targets):
            h_out, val_pred = forward_pass(pattern, mu, w, sigma)
            val_error += abs(target - np.sum(h_out * w))
        val_error /= val_patterns.shape[0]
        val_errors.append(val_error)

        if ES:
            if epoch > 1 and val_errors[-1][0] > val_errors[-2][0]:
                patience_counter += 1
            else:
                patience_counter = 0
            if patience_counter >= patience:
                print('Early stopping!!')
                return w

        print(f'EPOCH {epoch}\t| error {error[0]} | val error {val_errors[-1][0]}')

        if epoch % 10 == 0 and plot and not import_data:
            # clear_output(wait=False)
            pred = [forward_pass(x, mu, w, sigma)[1] for x in patterns]
            # fig = plt.figure()
            line.set_xdata(patterns)
            line.set_ydata(pred)
            ax.relim()
            ax.autoscale_view(True, True, True)
            fig.canvas.draw()
            plt.pause(0.01)
    return w


def forward_pass(pattern, mu, w, sigma):

    h_in = np.abs(mu - pattern)

    if h_in.shape[1] != 1:
        new_h_in = np.zeros((h_in.shape[0], 1))
        for row in range(h_in.shape[0]):
            for col in range(h_in.shape[1]):
                new_h_in[row] = np.sqrt(np.sum(h_in[row]**2))
        h_in = new_h_in

    h_out = phi(h_in, sigma)
    o_out = np.sum(w * h_out, axis=0)

    return h_out, o_out


def update_weights(target, h_out, w, lr):
    w += lr * (target - np.sum(h_out * w)) * h_out
    return w


def print_function(f, start=0, stop=2 * np.pi):
    x = np.linspace(start, start + stop, int(stop / 0.1))
    plt.figure()
    plt.plot(x, f)
    plt.show()


def euclidean_distance(a, b):
    return np.linalg.norm(a - b)


def get_patterns_indexes(patterns):
    return [i for i in range(patterns.shape[0])]


def get_first_n_argmins(list, n):
    return np.asarray(list).argsort()[:n]


def get_update_of_mean(lr, single_pattern, mu):
    return lr * (single_pattern - mu)


def competitive_learning(patterns, mu, lr_cl, n_nodes, more_winners, n_winners=1, max_epochs=10):
    for _ in range(max_epochs):
        patterns_idx = get_patterns_indexes(patterns)
        # shuffle pattern indexes
        np.random.shuffle(patterns_idx)
        for i in range(patterns.shape[0]):
            # at each iteration of CL a training vector is randomly selected from the data
            selected_pattern = patterns[patterns_idx[i]]
            # find the closest RBF unit
            dist_from_rbf_nodes = []
            for node_idx in range(n_nodes):
                dist_from_rbf_nodes.append(euclidean_distance(selected_pattern, mu[node_idx]))

            if more_winners:
                # n closest RBF units - winners
                nearest_rbf_nodes_idx = get_first_n_argmins(dist_from_rbf_nodes, n_winners)
                for winner_idx in nearest_rbf_nodes_idx:
                    mu[winner_idx] += get_update_of_mean(lr_cl, selected_pattern, mu[winner_idx])
            else:
                # single closest RBF unit - winner
                nearest_rbf_node_idx = np.argmin(dist_from_rbf_nodes)
                mu[nearest_rbf_node_idx] += get_update_of_mean(lr_cl, selected_pattern,
                                                               mu[nearest_rbf_node_idx])


def plot(patterns, targets, preds, lr, num_nodes, max_epochs,
         batch=False, cl=False, es=False, patience=None, MLP=False,
         lr_cl=None, epochs_cl=None, more_winners=False,
         import_data=False, val_patterns=None, val_targets=None,
         test_patterns=None, test_targets=None, centroids=None):

    patterns = np.array(patterns)
    targets = np.array(targets)
    preds = np.array(preds)


    if import_data:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        # ax.scatter(patterns[:, 0], patterns[:, 1], preds[:, 0], cmap='Greens')
        # ax.scatter(patterns[:, 0], patterns[:, 1], preds[:, 0], cmap='Greens')

        fig = plt.figure()
        plt.axis('off')
        ax = fig.add_subplot(projection='3d')
        X, Y, Z = patterns[:, 0], patterns[:, 1], preds[:, 0]

        ax.plot_surface(X, Y, Z)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()


    else:
        plt.figure()
        plt.plot(patterns, targets, label='True values')
        plt.plot(patterns, preds, label='Predictions')


    plt.legend()

    plt.xlabel('x')
    plt.ylabel('y')

    # title builder
    title = f''
    if MLP:
        title += 'MLP'
    else:
        if batch:
            title += 'RBF batch mode'
        else:
            if cl:
                title += f'RBF competitive'
            else:
                title += 'RBF seq mode'

    title += f' - hn {num_nodes} - epochs {max_epochs} - lr {lr}'

    if es:
        title += f' - es {es} - patience {patience}'
    if cl:
        title += f'\nlr_cl {lr_cl} - epochs_cl {epochs_cl} - more_winner {more_winners}'
    plt.title(title)
    plt.show()


def train_val_split(data, val_p, col_of_separation_patterns_targets):
    np.random.shuffle(data)
    val = data[:int(val_p * data.shape[0]), :]
    train = data[int(val_p * data.shape[0]):, :]
    return train[:, :col_of_separation_patterns_targets], train[:, col_of_separation_patterns_targets:], \
           val[:, :col_of_separation_patterns_targets], val[:, col_of_separation_patterns_targets:]
