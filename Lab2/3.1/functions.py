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


def residual_error(test_preds, test_targets):
    return sum(abs(test_preds-test_targets)) / test_preds.shape[0]

def train_seq(patterns, targets, w, max_epochs, sigma, mu, lr, plot, ES, val_patterns, val_targets, patience,
              import_data):

    if plot and not import_data:
        pred = [forward_pass(x, mu, w, sigma)[1] for x in patterns]
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        line, = ax.plot(patterns, pred)
        fig.canvas.draw()
        plt.show(block=False)

    val_errors = []
    train_errors = []
    patience_counter = 0
    for epoch in range(max_epochs):
        error = 0
        for pattern, target in zip(patterns, targets):
            h_out, _ = forward_pass(pattern, mu, w, sigma)
            w = update_weights(target, h_out, w, lr)
            error += abs(target - np.sum(h_out * w))
        error /= patterns.shape[0]
        train_errors.append(error)

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
    return w, train_errors, val_errors


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

    w += lr * (target - np.sum(h_out * w, axis=0)) * h_out
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
                # strategy to avoid dead units
                # n closest RBF units - winners
                nearest_rbf_nodes_idx = get_first_n_argmins(dist_from_rbf_nodes, n_winners)
                for winner_idx in nearest_rbf_nodes_idx:
                    mu[winner_idx] += get_update_of_mean(lr_cl, selected_pattern, mu[winner_idx])
            else:
                # single closest RBF unit - winner
                nearest_rbf_node_idx = np.argmin(dist_from_rbf_nodes)
                mu[nearest_rbf_node_idx] += get_update_of_mean(lr_cl, selected_pattern,
                                                               mu[nearest_rbf_node_idx])

def title_builder(MLP, batch, cl, num_nodes, max_epochs, lr,
                  es, patience, lr_cl, epochs_cl, more_winners, test, validation, learning_curves=False):
    # title builder
    title = f''

    if test:
        title += 'Test - '
    elif validation:
        title += 'Validation - '
    elif learning_curves:
        title += 'Learning curves - '
    else:
        title += 'Training - '

    if MLP:
        title += 'MLP'
    else:
        if batch:
            title += 'RBF batch mode'
        else:
            if cl:
                title += f'RBF seq competitive'
            else:
                title += 'RBF seq mode'

    title += f' - hn {num_nodes} - epochs {max_epochs} - lr {lr}'

    if es:
        title += f' - es {es} - patience {patience}'
    if cl:
        title += f'\nlr cl {lr_cl} - epochs cl {epochs_cl} - more winners {more_winners}'
    return title

def plot_test_results(test_patterns, test_targets, test_preds):
    # currently, it works only for sine and square function
    plt.figure()
    plt.title('test set')
    plt.plot(test_patterns, test_targets, label='True values - test set')
    plt.plot(test_patterns, test_preds, label='Predictions - test set')
    plt.legend()
    plt.show()

def plot(patterns, targets, preds, lr, num_nodes, max_epochs,
         batch=False, cl=False, es=False, patience=None, MLP=False,
         lr_cl=None, epochs_cl=None, more_winners=False,
         import_data=False, val_patterns=None, val_preds=None,
         centroids=None, test=False, validation=False):

    patterns = np.array(patterns)
    targets = np.array(targets)
    preds = np.array(preds)

    val_patterns = np.array(val_patterns)
    val_preds = np.array(val_preds)

    # angle, velocity
    # distance, height

    if import_data:
        for c in range(targets.shape[1]):
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.set_xlabel('Angle')
            ax.set_ylabel('Velocity')

            if c == 0:
                ax.set_zlabel('Distance')
            else:
                ax.set_zlabel('Height')

            ax.scatter(patterns[:, 0], patterns[:, 1], preds[:, c], label='Preds', c='k')
            ax.scatter(patterns[:, 0], patterns[:, 1], targets[:, c], label='True')
            ax.scatter(centroids[:, 0], centroids[:, 1], 0, s=100, marker='x', c='r')
            # if val_patterns is not None:
            #     ax.scatter(val_patterns[:, 0], val_patterns[:, 1], val_preds[:, c], label='Validation')
            plt.title(title_builder(MLP, batch, cl, num_nodes, max_epochs, lr,
                        es, patience, lr_cl, epochs_cl, more_winners, test, validation))
            plt.legend()
            plt.show()


    else:
        plt.figure()
        plt.plot(patterns, targets, label='True values')
        plt.plot(patterns, preds, label='Predictions')
        # if val_patterns is not None:
        #     plt.plot(val_patterns, val_preds, label='Validation')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.title(title_builder(MLP, batch, cl, num_nodes, max_epochs, lr,
                                es, patience, lr_cl, epochs_cl, more_winners, test, validation))
        plt.show()


def train_val_split(data, val_p, col_of_separation_patterns_targets):
    np.random.shuffle(data)
    val = data[:int(val_p * data.shape[0]), :]
    train = data[int(val_p * data.shape[0]):, :]
    return train[:, :col_of_separation_patterns_targets], train[:, col_of_separation_patterns_targets:], \
           val[:, :col_of_separation_patterns_targets], val[:, col_of_separation_patterns_targets:]


def mse(preds, targets):
    errors = preds - targets
    return np.sum(errors ** 2) / len(preds)


def get_continuous_predictions(mu, w, sigma, patterns):
    return [forward_pass(x, mu, w, sigma)[1] for x in patterns]


def get_discrete_predictions(mu, w, sigma, patterns):
    return [1 if forward_pass(x, mu, w, sigma)[1] >= 0 else -1 for x in patterns]


def plot_train_val(mse_errors_train, mse_errors_val, ballistic_data, lr, num_nodes, max_epochs,
         batch=False, cl=False, es=False, patience=None, MLP=False,
         lr_cl=None, epochs_cl=None, more_winners=False,
         import_data=False, val_patterns=None, val_preds=None,
         centroids=None, test=False, validation=False):
    if ballistic_data:
        mse_errors_train = np.mean(mse_errors_train, axis=1)
        mse_errors_val = np.mean(mse_errors_val, axis=1)

    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    fig, ax1 = plt.subplots()
    plt.title(title_builder(MLP, batch, cl, num_nodes, max_epochs, lr, es, patience, lr_cl, epochs_cl, more_winners,
                            test, validation, learning_curves=True))
    mse_line, = ax1.plot(mse_errors_train, color='red', label='Training MSE')
    ax2 = ax1.twinx()
    mse_line_val, = ax2.plot(mse_errors_val, color='blue', label='Validation MSE')
    ax2.legend(handles=[mse_line, mse_line_val])
    fig.tight_layout()
    plt.show()
