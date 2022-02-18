import numpy as np
import matplotlib.pyplot as plt
import random

n = 100
# mA = [1.0, 1.0]
# mB = [-1.0, -2.0]

mA = [2.0, 2.0]
mB = [-2.0, -3.0]

sigmaA = 0.5
sigmaB = 0.5
epochs = 4

np.random.seed(1)
random.seed(2)

delta = True
seq_delta = False

bias = -1


def get_data_points_coords():
    classA_1 = np.random.randn(1, n) * sigmaA + mA[0]
    classA_2 = np.random.randn(1, n) * sigmaA + mA[1]

    classB_1 = np.random.randn(1, n) * sigmaB + mB[0]
    classB_2 = np.random.randn(1, n) * sigmaB + mB[1]

    return classA_1, classA_2, classB_1, classB_2


def get_patterns(classA_1, classA_2, classB_1, classB_2, delta):
    target_1 = np.ones(shape=classA_1.shape)
    if delta:
        target_2 = -np.ones(shape=classB_1.shape)
    else:
        target_2 = np.zeros(shape=classB_1.shape)

    classA = [(classA_1[0][i], classA_2[0][i], target_1[0][i]) for i in range(len(classA_1[0]))]
    classB = [(classB_1[0][i], classB_2[0][i], target_2[0][i]) for i in range(len(classB_1[0]))]

    patterns = classA + classB
    random.shuffle(patterns)
    return patterns


def find_slope_interc(weights):
    # weights = [w1, w2,
    w0 = weights[0]
    w1 = weights[1]
    w2 = weights[2]

    m = - w0 / w1
    inter = - w2 / w1

    # inter = -b / w2
    # m = -(b / w2) / (b / w1)
    return inter, m


def plot_error(delta, all_tot_e):
    plt.figure(2, figsize=(10, 5))
    plt.title("learning curves")
    plt.xlim([1,70])
    for k, v in all_tot_e.items():
        plt.plot(np.arange(0, len(all_tot_e[k])), all_tot_e[k], label=str(k))
    plt.xlabel("epochs")
    if delta:
        plt.ylabel("error")
    else:
        plt.ylabel("misclassified samples")
    plt.legend()
    plt.show()


def plottalo(weights_matrix, classA_1, classA_2, classB_1, classB_2, epoch, lr):
    plt.figure(1, figsize=(10, 5))
    plt.title(f"epoch {str(epoch)} - lr {lr}")

    print(classA_1)
    x_lim_max = max(max(classA_1[0]) , max(classB_1[0]))
    x_lim_min = min(min(classA_1[0]), min(classB_1[0]))

    y_lim_max = max(max(classA_2[0]) , max(classB_2[0]))
    y_lim_min = min(min(classA_2[0]), min(classB_2[0]))

    plt.xlim([x_lim_min - 1, x_lim_max + 1])
    plt.ylim([y_lim_min - 1, y_lim_max + 1])
    inter, m = find_slope_interc(weights_matrix)

    x = np.linspace(-1000, 1000)
    y = m * x - inter * bias

    plt.plot(x, y, '-')

    plt.scatter(classA_1[0], classA_2[0], c='r')
    plt.scatter(classB_1[0], classB_2[0], c='b')
    plt.show()



def main(learning_rates):
    classA_1, classA_2, classB_1, classB_2 = get_data_points_coords()

    patterns = get_patterns(classA_1, classA_2, classB_1, classB_2, delta)

    targets = [patterns[i][2] for i in range(len(patterns))]
    targets = np.array(targets, dtype=int)


    patterns = [(patterns[i][0], patterns[i][1], bias) for i in range(len(patterns))]
    patterns = np.transpose(np.array(patterns))

    init_weights = np.random.normal(loc=0.0, scale=1.0, size=(1, 3))[0]

    all_tot_e = {}
    for lr in learning_rates:
        weights = np.copy(init_weights)
        all_tot_e[lr] = []
        if delta:
            y_prime = np.dot(weights, patterns)
            all_e = y_prime - targets
            tot_e = np.sum(all_e ** 2)
            all_tot_e[lr].append(tot_e)

        for epoch in range(epochs):
            if delta:
                # delta rule
                if seq_delta:
                    # online delta rule
                    tot_e = 0
                    y_prime = []
                    for t in range(len(patterns[0])):
                        y_t = np.dot(weights, patterns[:, t])
                        y_prime.append(y_t)
                        e = y_t - targets[t]
                        tot_e += e**2
                        weights -= lr * patterns[:, t] * e
                else:
                    # batch delta rule
                    y_prime = np.dot(weights, patterns)
                    all_e = y_prime - targets
                    tot_e = np.sum(all_e**2)
                    d_w = - lr * all_e @ np.transpose(patterns)
                    weights += d_w
                all_tot_e[lr].append(tot_e)
            else:
                # perceptron learning rule
                y_pred_perceptron_rule = []
                for t in range(len(patterns[0])):
                    y_t = np.dot(weights, patterns[:, t])
                    if y_t >= 0:
                        label_t = 1
                    else:
                        label_t = 0
                    y_pred_perceptron_rule.append(label_t)
                    e = label_t - targets[t]
                    weights -= lr * patterns[:, t] * e
            # plot data points and decision boundary of first and last iteration
            if epoch == 0 or epoch == epochs - 1:
                plottalo(weights, classA_1, classA_2, classB_1, classB_2, epoch, lr)

            # STOPPING RULES
            if not delta:
                # print(np.sum(y_pred_perceptron_rule == targets))
                if np.sum(y_pred_perceptron_rule == targets) == n * 2:
                    plottalo(weights, classA_1, classA_2, classB_1, classB_2, epoch, lr)
                    print(f'correctly classified samples --> {np.sum(y_pred_perceptron_rule == targets)}')
                    print(f'epochs --> {epoch}')
                    all_tot_e[lr].append(2*n - np.sum(y_pred_perceptron_rule == targets))
                    break
            else:
                y_prime = np.asarray(y_prime)
                pred = np.where(y_prime >= 0, 1, -1)
                if np.sum(pred == targets) == n * 2:
                    plottalo(weights, classA_1, classA_2, classB_1, classB_2, epoch, lr)
                    print(f'correctly classified samples --> {np.sum(pred == targets)}')
                    print(f'epochs --> {epoch}')
                    print(f'error --> {tot_e}')
                    all_tot_e[lr].append(tot_e)
                    break

    plot_error(delta, all_tot_e)


if __name__ == '__main__':
    # learning_rates = [0.0005, 0.001, 0.01, 0.1]
    learning_rates = [0.0001, 0.001]
    main(learning_rates)
