import numpy as np

weights = np.array([
    [1, 0, -1],
    [0, -3, -1]
], dtype=np.float64)
bias = np.array([[0], [1]], dtype=np.float64)
patterns = np.array([
    [1, -1, 1],
    [1, -1, 0],
    [-1, 2, 1]
], dtype=np.float64)
targets = np.array([
    [-1, 1, 1],
    [-1, 1, -1]
], dtype=np.float64)

y_prime = np.dot(weights, patterns) + bias
all_e = y_prime - targets
tot_e = np.sum(all_e**2)
d_w = - 0.1 * all_e @ np.transpose(patterns)
weights += d_w
print(weights)
y_prime = np.dot(weights, patterns) + bias
all_e = y_prime - targets
tot_e = np.sum(all_e**2)
d_w = - 0.1 * all_e @ np.transpose(patterns)
weights += d_w
print(weights)
y_prime = np.dot(weights, patterns) + bias
print(np.where(y_prime >= 0, 1, -1))