import json
import pdb
import codecs
import pdb
import pandas as pd

def get_data():
    statements = []
    X, y = pd.read_csv('data/SD_dataset_FINAL.csv')
    #pdb.set_trace()
    return X


if __name__=="__main__":
    tweets = get_data()
    males, females = {}, {}
    with open('./tweet_data/males.txt') as f:
        males = set([w.strip() for w in f.readlines()])
    with open('./tweet_data/females.txt') as f:
        females = set([w.strip() for w in f.readlines()])

    males_c, females_c, not_found = 0, 0, 0
    for t in tweets:
        if t['name'] in males:
            males_c += 1
        elif t['name'] in females:
            females_c += 1
        else:
            not_found += 1
    pdb.set_trace()
