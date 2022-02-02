import numpy as np
import matplotlib.pyplot as plt

from numpy.random import multivariate_normal

n = 100
m_A = [1.0, 0.3]
m_B = [0.0, -0.1]

sigma_A = 0.2
sigma_B = 0.3

classA_1 = multivariate_normal(m_A, [[sigma_A,0]], n)
# classA_2 = multivariate_normal()