import numpy as np


def init_weights(size):
    return np.random.random(size=size)


def get_votes(path, shape):
    data = np.genfromtxt(path, delimiter=',')
    data = data.reshape(shape)
    return data


def get_parties(path):
    party = np.genfromtxt(path, comments="%")
    return party


def get_genders(path):
    sex = np.genfromtxt(path, comments="%")
    return sex


def get_districts(path):
    districts = np.genfromtxt(path, comments="%")
    return districts


def get_names(path):
    names = np.loadtxt(path, dtype=str, delimiter='\n')
    for i in range(len(names)):
        names[i] = names[i].replace("'", '')
    return names

