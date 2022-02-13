import numpy as np
import fileinput


def get_city_matrix(path='data/cities.dat', header_lines_to_skip=4):
    for line in fileinput.input(path, inplace=True):
        print('{}'.format(line.replace(';', '')), end='')
    city_matrix = np.genfromtxt(path, delimiter=',', skip_header=header_lines_to_skip)
    return city_matrix
