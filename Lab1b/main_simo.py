import numpy as np
import matplotlib.pyplot as plt

from numpy.random import multivariate_normal
from numpy.random import normal

n = 100
m_A = [1.0, 0.3]
m_B = [0.0, -0.1]

sigma_A = 0.2
sigma_B = 0.3

bias = 1

def get_patterns():
    # create class A (disjoint) and B, with specified global means and cov (diagonal)
    # return classes with bias coordinate
    classA_1 = multivariate_normal(m_A, [[sigma_A,0],[0, sigma_A]], int(n * 0.5))
    classA_2 = multivariate_normal([-m_A[0],-m_A[1]], [[sigma_A,0],[0, sigma_A]], int(n * 0.5))

    classA = np.concatenate((classA_1,classA_2))
    classB = multivariate_normal(m_B, [[sigma_B,0],[0, sigma_B]], n)


    plt.scatter(classA[:,0], classA[:,1])
    plt.scatter(classB[:,0], classB[:,1])

    patterns = np.array([[[x[0], x[1], bias] for x in classA] + [[x[0], x[1], bias] for x in classB]])
    targets = np.array([[1 for x in classA] + [-1 for x in classB]])

    return patterns, targets



# w_0 = normal(0, 1, )
