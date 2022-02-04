import numpy as np
import tensorflow as tf

from keras import initializers
from keras import optimizers
from keras import regularizers

from keras.models import Sequential
from keras.layers import Input, Dense
from keras.callbacks import EarlyStopping

from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

import pandas as pd

train_val_p = 0.8
BIAS = 0.
BATCH_SIZE = 32
# HIDDEN_NODES = [3, 6]
EPOCHS = 10000
# LR = 0.01
# l2 = 0.0002
ES = True

NOISE = False
SIGMA = 0.15

MAX_ITER = 10


def mackey_glass_generator(n_samples=1600, beta=0.2, gamma=0.1, n=10, tau=25):
    x0 = 1.5
    x_values = [x0]

    for t in range(n_samples):
        if (t - tau) < 0:
            past_x = 0
        else:
            past_x = x_values[t - tau]
        next_t = x_values[-1] + (beta * past_x / (1 + pow(past_x, n))) - gamma * x_values[-1]
        x_values.append(next_t)

    return x_values

def data_from_mackey_glass(x, idx_0=301, idx_final=1500):
    data = []
    labels = []
    for t in range(idx_0, idx_final + 1):
        data.append([x[t-20], x[t-15], x[t-10], x[t-5], x[t]])
        labels.append(x[t+5])

    return np.array(data), np.array(labels)

def train_test_val_split(data,labels, train_p):
    '''
    Last 200 for test, then we choose train/validation proportion
    '''
    train = data[:int(1000 * train_p), :]
    train_labels = labels[:int(1000 * train_p)]

    val = data[int(1000 * train_p):1000, :]
    val_labels = labels[int(1000 * train_p):1000]

    test = data[1000:, :]
    test_labels = labels[1000:]

    return train, train_labels, val, val_labels, test, test_labels

def add_noise(x, sigma, idx_0=301, idx_final=301+int(1000 * train_val_p)):
    for t in range(idx_0, idx_final+1):
        x[t] += np.random.normal(scale=sigma)
    return x

def plot_time_series(x):
    plt.plot(x)
    plt.show()

def preds_accuracy_plot(y_test, preds, HIDDEN_NODES):
    mse = mean_squared_error(y_test, preds)
    plt.title(f'MSE: {mse:.3f} - HN: {HIDDEN_NODES} - LR: {LR} - ES: {ES} - L2 {l2}')
    plt.plot(y_test)
    plt.plot(preds)
    plt.show()


def main(a, b, LR, l2, my_dicts):
    HIDDEN_NODES = [a, b]
    mses = []
    for iter in range(MAX_ITER):
        x = mackey_glass_generator()
        if NOISE:
            x = add_noise(x, SIGMA)
        # plot_time_series(x)

        data, labels = data_from_mackey_glass(x)
        x_train, y_train, x_val, y_val, x_test, y_test = train_test_val_split(data, labels, train_val_p)

        # plot_time_series(x)



        model = Sequential()


        model.add(Input(shape=(5,)))
        for i in range(len(HIDDEN_NODES)):
            model.add(Dense(
                HIDDEN_NODES[i],
                activation='sigmoid',
                use_bias=True,
                kernel_initializer=initializers.initializers_v2.RandomNormal(mean=0., stddev=1.),
                bias_initializer=initializers.initializers_v2.Constant(BIAS),
                kernel_regularizer=regularizers.l2(l2=l2),
                bias_regularizer=regularizers.l2(l2=l2),
                ))
        model.add(tf.keras.layers.Dense(1, activation='linear'))

        # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        #     initial_learning_rate=1e-2,
        #     decay_steps=10000,
        #     decay_rate=0.9)

        optimizer = optimizers.gradient_descent_v2.SGD(learning_rate=LR)

        model.compile(
            loss='mse',
            optimizer=optimizer
            )


        callbacks= []
        if ES:
            es = EarlyStopping(monitor='val_loss', patience=10, min_delta=0.00001)
            callbacks.append(es)


        model.fit(x_train, y_train,
                  batch_size=BATCH_SIZE,
                  epochs=EPOCHS,
                  verbose='auto',
                  callbacks=callbacks,
                  validation_data=(x_val, y_val),
                  workers=8)


        preds = model.predict(x_test)
        mse = mean_squared_error(y_test, preds)
        preds_accuracy_plot(y_test, preds, HIDDEN_NODES)

        mses.append(mse)
    dic = {'hidden_nodes': HIDDEN_NODES, 'lr': LR, 'l2': l2}
    dic['mse_mean'] = np.mean(mses)
    dic['mse_std'] = np.std(mses)

    my_dicts.append(dic)


if __name__ == '__main__':
    # main()

    # GridSearch


    nodes_first = [3, 4, 5]
    nodes_second = [2, 4, 6]
    LRs = [0.1, 0.01, 0.05, 0.001]
    l2s = [0.001, 0.0001, 0.0002, 0.0005]


    # mse, HIDDEN_NODES, LR, l2, ES
    my_dicts = []

    for a in nodes_first:
        for b in nodes_second:
            for LR in LRs:
                for l2 in l2s:
                    main(a, b, LR, l2, my_dicts)


    scores_df = pd.DataFrame(my_dicts)
    scores_df.to_csv('mse_score.csv')
    print(scores_df)
