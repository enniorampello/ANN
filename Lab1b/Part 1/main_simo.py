import numpy as np
from numpy.random import multivariate_normal, normal
import matplotlib.pyplot as plt

n = 100
m_A = [1.0, 0.3]
m_B = [0.0, -0.1]

sigma_A = 0.2
sigma_B = 0.3

HIDDEN_NODES_LIST = [7]#[i+2 for i in range(20)]
bias = 1
learning_rate = 0.001
n_epochs = 1000
# set to 1 to remove data according to the percentages
# or set to 2 for removing data according to the third point in the assignment
val = 2
perc_A = 0.5
perc_B = 0.

batch = True
np.random.seed(2)


def f(x):
    return (2 / (1 + np.exp(-x))) - 1


def f_prime(x):
    return ((1 + f(x)) * (1 - f(x))) * 0.5


def get_patterns(val, perc_A=0.25, perc_B=0.25):
    # create class A (disjoint) and B, with specified global means and cov (diagonal)
    # return classes with bias coordinate
    patterns_val = None
    targets_val = None

    classA_1 = multivariate_normal(m_A, [[sigma_A ** 2, 0], [0, sigma_A ** 2]], int(n * 0.5))
    classA_2 = multivariate_normal([-m_A[0], m_A[1]], [[sigma_A ** 2, 0], [0, sigma_A ** 2]], int(n * 0.5))
    classA = np.concatenate((classA_1, classA_2))

    classB = multivariate_normal(m_B, [[sigma_B ** 2, 0], [0, sigma_B ** 2]], n)

    if val == 0:
        patterns = np.array([[x[0], x[1], bias] for x in classA] + [[x[0], x[1], bias] for x in classB]).transpose()
        targets = np.array([1 for x in classA] + [-1 for x in classB])
    elif val == 1:
        np.random.shuffle(classA)
        classA_train = classA[int(perc_A * classA.shape[0]):, :]
        classA_val = classA[:int(perc_A * classA.shape[0]), :]
        np.random.shuffle(classB)
        classB_train = classB[int(perc_B * classB.shape[0]):, :]
        classB_val = classB[:int(perc_B * classB.shape[0]), :]

        patterns = np.array(
            [[x[0], x[1], bias] for x in classA_train] + [[x[0], x[1], bias] for x in classB_train]).transpose()
        targets = np.array([1 for x in classA_train] + [-1 for x in classB_train])
        patterns_val = np.array(
            [[x[0], x[1], bias] for x in classA_val] + [[x[0], x[1], bias] for x in classB_val]).transpose()
        targets_val = np.array([1 for x in classA_val] + [-1 for x in classB_val])
    elif val == 2:
        classA_sx = np.array([x for x in classA if x[0] < 0])
        classA_dx = np.array([x for x in classA if x[0] > 0])

        np.random.shuffle(classA_sx)
        classA_sx_train = classA_sx[int(0.2 * classA_sx.shape[0]):, :]
        classA_sx_val = classA_sx[:int(0.2 * classA_sx.shape[0]), :]

        np.random.shuffle(classA_dx)
        classA_dx_train = classA_dx[int(0.8 * classA_dx.shape[0]):, :]
        classA_dx_val = classA_dx[:int(0.8 * classA_dx.shape[0]), :]

        classA_train = np.concatenate((classA_sx_train, classA_dx_train))
        classA_val = np.concatenate((classA_sx_val, classA_dx_val))

        patterns = np.array(
            [[x[0], x[1], bias] for x in classA_train] + [[x[0], x[1], bias] for x in classB]).transpose()
        targets = np.array([1 for x in classA_train] + [-1 for x in classB])

        patterns_val = np.array([[x[0], x[1], bias] for x in classA_val]).transpose()
        targets_val = np.array([1 for x in classA_val])
        pass

    return patterns, targets, patterns_val, targets_val


def forward_pass(patterns, w, v):
    h_in = w @ patterns
    h_out = f(h_in)
    o_in = v @ h_out
    o_out = f(o_in)
    return h_in, h_out, o_in, o_out

def forward_pass_seq(patterns, w, v):
    h_in = w @ patterns
    h_out = f(h_in)
    o_in = v @ h_out
    o_out = f(o_in)
    return h_in, h_out, o_in, o_out

def save_errors(o_out, targets, MSE_errors, miscl_error):
    MSE_errors.append(MSE(o_out, targets))
    miscl_error.append(misclass_rate(o_out, targets))


def backward_pass(v, targets, h_in, o_out, o_in, hidden_nodes):
    delta_o = np.multiply(np.subtract(o_out, targets), f_prime(o_in))
    v = v.reshape(1, hidden_nodes)
    delta_o = delta_o.reshape(1, delta_o.shape[1])
    delta_h = np.multiply((v.transpose() @ delta_o), f_prime(h_in))
    return delta_h, delta_o

def backward_pass_seq(v, targets, h_in, o_out, o_in):
    delta_o = (o_out - targets) * f_prime(o_in)
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

def weight_update_seq(weights, inputs, delta, lr, momentum=False, alpha=0.9, d_old=None):
    if momentum:
        print("momentum")
        d = (d_old * alpha) - (delta * np.transpose(inputs)) * (1 - alpha)
    else:
        try:
            inputs = inputs.reshape(hidden_nodes, 1)
        except:
            inputs = inputs.reshape(3, 1)

        try:
            delta = delta.reshape(hidden_nodes, 1)
        except:
            delta = delta.reshape(1, 1)

    d = delta @ inputs.transpose()
    weights -= np.multiply(d, lr)
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
    return error_rate / len(preds)


def plot_errors(MSE_errors, miscl_errors):
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    fig, ax1 = plt.subplots()
    mse_line, = ax1.plot(MSE_errors, color='red', label='MSE')
    ax2 = ax1.twinx()
    miscl_line, = ax2.plot(miscl_errors, color='blue', label='Misclassification rate')
    ax2.legend(handles=[mse_line, miscl_line])
    fig.tight_layout()
    plt.show()


def plot_train_val(MSE_errors_train, MSE_errors_val):
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    fig, ax1 = plt.subplots()
    mse_line, = ax1.plot(MSE_errors_train, color='red', label='Training MSE')
    ax2 = ax1.twinx()
    mse_line_val, = ax2.plot(MSE_errors_val, color='blue', label='Validation MSE')
    ax2.legend(handles=[mse_line, mse_line_val])
    fig.tight_layout()
    plt.show()

    # plt.rcParams["figure.figsize"] = [7.00, 3.50]
    # plt.rcParams["figure.autolayout"] = True
    # fig, ax = plt.subplots()
    # mse_line, = ax.plot(MSE_errors_train, color='red', label='Training MSE')
    # mse_line_val, = ax.plot(MSE_errors_val, color='blue', label='Validation MSE')
    # ax.legend(handles=[mse_line, mse_line_val])
    # fig.tight_layout()
    # plt.show()


def plot_boundary(patterns, targets, w, v, MSE, miscl, MSE_val=None, miscl_val=None, hn=0):
    x_min = min(patterns[0, :]) - 1
    x_max = max(patterns[0, :]) + 1
    y_min = min(patterns[1, :]) - 1
    y_max = max(patterns[1, :]) + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    input = np.array([[x[0], x[1], bias] for x in np.c_[xx.ravel(), yy.ravel()]])
    _, _, _, mesh_preds = forward_pass(input.transpose(), w, v)
    Z = np.where(mesh_preds > 0, 1, -1)[0]
    Z = Z.reshape(xx.shape)
    fig, ax = plt.subplots()
    if MSE_val is None:
        ax.set_title(f'MSE:{MSE:.2f} - miscl:{miscl:.2f} - hn:{hn} - lr:{learning_rate} - epochs:{n_epochs}')
    else:
        ax.set_title(
            f'MSE:{MSE[-1]:.2f} - miscl:{miscl:.2f} - MSE val:{MSE_val:.2f} - miscl val:{miscl_val:.2f} \nhn:{hidden_nodes} - lr:{learning_rate} - epochs:{n_epochs}')
    ax.contourf(xx, yy, Z, cmap=plt.cm.Paired)

    # Plot also the training points
    ax.scatter(patterns[0, :], patterns[1, :], c=targets)

    plt.show()


def main():
    MSEs = []
    miscls = []
    if val > 0:
        val_MSEs = []
        val_miscl = []
    patterns, targets, patterns_val, targets_val = get_patterns(val, perc_A=perc_A, perc_B=perc_B)

    min_MSE = 1000
    hn = 0
    it = 0

    for hidden_nodes in HIDDEN_NODES_LIST:
        partial_MSE = []
        partial_miscl = []
        if val > 0:
            partial_val_MSE = []
            partial_val_miscl = []

        for x in range(1, 11):

            w = normal(0, 1, [hidden_nodes, 3])
            v = normal(0, 1, hidden_nodes).reshape(1, hidden_nodes)

            dw = 0
            dv = 0

            MSE_errors = []
            MSE_errors_val = []
            miscl_errors = []
            miscl_errors_val = []

            if batch:
                for i_epoch in range(n_epochs):
                    h_in, h_out, o_in, o_out = forward_pass(patterns, w, v)
                    save_errors(o_out, targets, MSE_errors, miscl_errors)

                    if val > 0:
                        _, _, _, o_out_val = forward_pass(patterns_val, w, v)
                        save_errors(o_out_val, targets_val, MSE_errors_val, miscl_errors_val)

                    print(f"EPOCH {i_epoch:4d} | training_mse = {MSE(o_out, targets):4.2f} |")

                    delta_h, delta_o = backward_pass(v, targets, h_in, o_out, o_in, hidden_nodes)

                    w, dw = weight_update(w, patterns, delta_h, lr=learning_rate, momentum=False, d_old=dw)
                    v, dv = weight_update(v, h_out, delta_o, lr=learning_rate, momentum=False, d_old=dv)
            else:
                for i_epoch in range(n_epochs):
                    print(f"EPOCH {i_epoch:4d}")
                    for i in range(patterns.shape[1]):
                        h_in, h_out, o_in, o_out = forward_pass_seq(patterns[:, i], w, v)

                        # print(f"EPOCH {i_epoch:4d} | training_mse = {MSE(o_out, targets):4.2f} |")

                        delta_h, delta_o = backward_pass_seq(v, targets[i], h_in, o_out, o_in)
                        w, dw = weight_update_seq(w, patterns[:, i], delta_h, lr=learning_rate, momentum=False, d_old=dw)
                        v, dv = weight_update_seq(v, h_out, delta_o, lr=learning_rate, momentum=False, d_old=dv)

                    _, _, _, o_out = forward_pass_seq(patterns, w, v)
                    save_errors(o_out, targets, MSE_errors, miscl_errors)
                    if val > 0:
                        _, _, _, o_out_val = forward_pass(patterns_val, w, v)
                        save_errors(o_out_val, targets_val, MSE_errors_val, miscl_errors_val)
            # if MSE_errors[-1] < min_MSE:
            #     it = x
            #     hn = hidden_nodes
            partial_MSE.append(MSE_errors[-1])
            partial_miscl.append(miscl_errors[-1]*100.)
            if val > 0:
                partial_val_MSE.append(MSE_errors_val[-1])
                partial_val_miscl.append(miscl_errors_val[-1] * 100.)
        MSEs.append(np.mean(np.array(partial_MSE)))
        miscls.append(np.mean(partial_miscl))
        if val > 0:
            val_MSEs.append(np.mean(np.array(partial_val_MSE)))
            val_miscl.append(np.mean(partial_val_miscl))
    print(MSEs)
    print(miscls)
    print(val_MSEs)
    print(val_miscl)
    # plt.figure()
    # plt.title('MSE and miscl. rate given the number of nodes in the hidden layer')
    # plt.xlabel('Number of nodes')
    # plt.ylabel('MSE and miscl. rate')
    # plt.xticks(np.arange(2, 23))
    # plt.plot(np.arange(2, len(MSEs)+2), MSEs, label='MSE')
    # plt.plot(np.arange(2, len(MSEs)+2), miscls, label='Miscl. rate %')
    # plt.legend()
    # plt.show()




if __name__ == '__main__':
    main()
