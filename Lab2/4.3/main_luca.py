import numpy as np
from functions import *

# data shapes
VOTES_SHAPE = (349, 31)

# data paths
VOTES_PATH = "../4.1/data/votes.dat"
PARTIES_PATH = "../4.1/data/mpparty.dat"
GENDERS_PATH = "../4.1/data/mpsex.dat"
DISTRICTS_PATH = "../4.1/data/mpdistrict.dat"
NAMES_PATH = "../4.1/data/mpnames.txt"


def main():
    # load data
    # 0 = no-vote; 1 = yes-vote; 0.5 = missing vote
    # each row -is one mp; each col is one vote
    votes = get_votes(VOTES_PATH, VOTES_SHAPE)
    # each elem of parties, genders, names is the party, gender, name of the correspondent mp

    # 0=no party, 1='m', 2='fp', 3='s', 4='v', 5='mp', 6='kd', 7='c'
    parties = get_parties(path=PARTIES_PATH)
    # Male 0, Female 1
    genders = get_genders(path=GENDERS_PATH)
    districts = get_districts(path=DISTRICTS_PATH)
    names = get_names(path=NAMES_PATH)


if __name__ == '__main__':
    main()
