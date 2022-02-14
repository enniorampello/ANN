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
    # 0 = no-vote; 1 = yes-vote; 0.5 = missing vote
    votes = get_votes(path=VOTES_PATH)
    # 0=no party, 1='m', 2='fp', 3='s', 4='v', 5='mp', 6='kd', 7='c'
    parties = get_parties(path=PARTIES_PATH)
    # Male 0, Female 1
    genders = get_genders(path=GENDERS_PATH)
    districts = get_districts(path=DISTRICTS_PATH)
    names = get_names(path=NAMES_PATH)


if __name__ == '__main__':
    main()
