import numpy as np


def init_weights(size):
    return np.random.random(size=size)


def get_votes(path='../4.1/data/votes.dat'):
    data = np.genfromtxt(path, delimiter=',')
    return data


def get_parties(path='../4.1/data/mpparty.dat'):
    party = np.genfromtxt(path, comments="%")
    return party


def get_genders(path='../4.1/data/mpsex.dat'):
    sex = np.genfromtxt(path, comments="%")
    return sex


def get_districts(path='../4.1/data/mpdistrict.dat'):
    districts = np.genfromtxt(path, comments="%")
    return districts


def get_names(path='../4.1/data/mpnames.txt'):
    names = np.loadtxt(path, dtype=str, delimiter='\n')
    for i in range(len(names)):
        names[i] = names[i].replace("'", '')
    return names

