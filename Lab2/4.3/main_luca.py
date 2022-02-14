import numpy as np
from functions import *

# data paths
VOTES_PATH = "../4.1/data/votes.dat"
PARTIES_PATH = "../4.1/data/votes.dat"
GENDERS_PATH = "../4.1/data/mpsex.dat"
DISTRICTS_PATH = "../4.1/data/mpdistrict.dat"
NAMES_PATH = "../4.1/data/mpnames.txt"


def main():
    # load data
    votes = get_votes(path=VOTES_PATH)
    parties = get_parties(path=PARTIES_PATH)
    genders = get_genders(path=GENDERS_PATH)
    districts = get_districts(path=DISTRICTS_PATH)
    names = get_names(path=NAMES_PATH)


if __name__ == '__main__':
    main()
