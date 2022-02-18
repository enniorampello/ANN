import numpy as np

x1 = np.array([-1, -1, 1, -1, 1, -1, -1, 1])
x2 = np.array([-1, -1, -1, -1, -1, 1, -1, -1])
x3 = np.array([-1, 1, 1, -1, -1, 1, -1, 1])

patterns = np.vstack((x1, x2, x3))
w = patterns.T @ patterns
print(patterns)

energy_x1 = - w @ x1 @ x1
energy_x2 = - w @ x2 @ x2
energy_x3 = - w @ x3 @ x3
print(energy_x1, energy_x2, energy_x3)

x1d = np.array([1, -1, 1, -1, 1, -1, -1, 1])
x2d = np.array([1, 1, -1, -1, -1, 1, -1, -1])
x3d = np.array([1, 1, 1, -1, 1, 1, -1, 1])

energy_x1d = - w @ x1d @ x1d
energy_x2d = - w @ x2d @ x2d
energy_x3d = - w @ x3d @ x3d
print(energy_x1d, energy_x2d, energy_x3d)
