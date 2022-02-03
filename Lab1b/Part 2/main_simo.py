import numpy as np
import tensorflow as tf

def mackey_glass_generator(n_samples = 1600, beta=0.2, gamma=0.1, n=10, tau=25):
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


x = mackey_glass_generator()
data, labels = data_from_mackey_glass(x)
train, train_labels, val, val_labels, test, test_labels = train_test_val_split(data, labels, 0.8)



model = tf.keras.Sequential()

hidden_nodes = 5

model.add(tf.keras.layers.Dense(hidden_nodes,
                                activation=tf.keras.activations.sigmoid,
                                input_shape=(5,)))

model.add(tf.keras.layers.Dense(1, activation=tf.nn.relu))


es = tf.keras.callbacks.EarlyStopping(monitor='val_loss')
# model.compile(loss=)
model.summary()