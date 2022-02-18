import numpy as np
import matplotlib.pyplot as plt
import random

n = 100
mA = [1.0, 0.5]
mB = [-3.0, -3.0]
sigmaA = 0.5
sigmaB = 0.5
lr = 0.001
epochs = 20

np.random.seed(5)
random.seed(5)

delta = True

classA_1 = np.random.randn(1, n) * sigmaA + mA[0]
classA_2 = np.random.randn(1, n) * sigmaA + mA[1]
classB_1 = np.random.randn(1, n) * sigmaB + mB[0]
classB_2 = np.random.randn(1, n) * sigmaB + mB[1]

target_1 = np.ones(shape=classA_1.shape)
if delta:
    target_2 = -np.ones(shape=classB_1.shape)
else:
    target_2 = np.zeros(shape=classB_1.shape)

classA = [(classA_1[0][i], classA_2[0][i], target_1[0][i]) for i in range(len(classA_1[0]))]
classB = [(classB_1[0][i], classB_2[0][i], target_2[0][i]) for i in range(len(classB_1[0]))]

patterns = classA + classB
random.shuffle(patterns)
patterns = np.transpose(np.array(patterns))

targets = [patterns[i][2] for i in range(len(patterns))]
targets = np.array(targets, dtype=int)

weights = np.random.normal(loc=0.0, scale=1.0, size=(1, 3))
weights = weights[0]

for epoch in range(epochs):
    y_prime = np.dot(weights, patterns)
    if delta:
        weights -= lr * (np.dot(weights, patterns) - targets) @ np.transpose(targets)

    plt.figure(1, figsize=(10, 5))
    plt.xlim([-5, 5])
    plt.ylim([-4, 3])
    w0 = weights[0]
    w1 = weights[1]
    w2 = weights[2]

    m = - w0 / w1
    inter = - w2 / w1

    x = np.linspace(-1000, 1000)
    y = m * x - inter

    plt.plot(x, y, '-')

    plt.scatter(classA_1[0], classA_2[0], c='r')
    plt.scatter(classB_1[0], classB_2[0], c='b')
    plt.show()