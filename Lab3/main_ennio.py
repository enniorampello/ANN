import itertools
import numpy as np
import pandas as pd
from functions import *

np.random.seed(0)

x1 = np.array([-1, -1, 1, -1, 1, -1, -1, 1])
x2 = np.array([-1, -1, -1, -1, -1, 1, -1, -1])
x3 = np.array([-1, 1, 1, -1, -1, 1, -1, 1])
# # 
patterns = np.vstack((x1, x2, x3))
w = patterns.T @ patterns
# energy_x1 = - w @ x1 @ x1
# energy_x2 = - w @ x2 @ x2
# energy_x3 = - w @ x3 @ x3
# print(energy_x1, energy_x2, energy_x3)
# 
x1d = np.array([1, -1, 1, -1, 1, -1, -1, 1])
x2d = np.array([1, 1, -1, -1, -1, 1, -1, -1])
x3d = np.array([1, 1, 1, -1, 1, 1, -1, 1])
# 
print(x1)
print(synch_update(x1d, w))
print(x2)
print(synch_update(x2d, w))
print(x3)
print(synch_update(x3d, w))

# patterns = itertools.product((-1, 1), repeat=8)
# count = 0
# for i, pattern in enumerate(patterns):
#     if (pattern == synch_update(pattern, w)).all():
#         count += 1
# print(count)

# x2dd = np.array([1, 1, 1, 1, 1, 1, -1, -1])
# print(x2)
# print(synch_update(x2dd, w))

# energy_x1d = asynch_update(x1d, w, energy=True)
# energy_x2d = asynch_update(x2d, w, energy=True)
# energy_x3d = asynch_update(x3d, w, energy=True)
# print(energy_x1d, energy_x2d, energy_x3d)

# w = np.random.randn(8, 8)
# w = 0.5 * (w + w.T)
# x_rand = np.random.choice([-1, 1], size=8)
# asynch_update(x_rand, w, energy=True)

# BIASES_SPARSE = np.linspace(0, 10, 10) * 0.1
# ACTIVITIES = [0.1, 0.05, 0.01]
# 
# 
# bias_act_max = pd.DataFrame(index=ACTIVITIES, columns=BIASES_SPARSE)
# bias_act_max.loc[0.1, 0.0] = 'suca'
# print(bias_act_max)