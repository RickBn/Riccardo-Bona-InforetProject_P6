import json
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from string import punctuation
from nltk.corpus import stopwords
import spacy
import nltk
import numpy as np

D = pd.read_json("data/ggt2.json")
titles = D['title']
desc = D['description']
categ = D['categories']


import pickle

with open ('data\games.json', 'rb') as fp:
    games = pickle.load(fp)

categories = list()
for i in range(len(D)):
    for c in D['categories'][i]:
        if c not in categories and c != "":
            categories.append(c)

for i in range(len(D)):
    for c in D['categories'][i]:
        if c == "":
            D['categories'][i].clear()

categ = D['categories']

from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

word = ['economic']

desc_model = {}
desc_model = Word2Vec(desc, min_count=1, window=2)
print([x[0] for x in desc_model.wv.most_similar(positive=word)])

categ_model = {}

categ_model = Word2Vec(categ, min_count=1, window=2)


print([x[0] for x in categ_model.wv.most_similar(positive=word)])
german = [x[0] for x in categ_model.wv.most_similar(positive=word)]
german.append('economic')

word = ['fantasy']
print([x[0] for x in categ_model.wv.most_similar(positive=word)])
american = [x[0] for x in categ_model.wv.most_similar(positive=word)]
american.append('fantasy')

D['rating_avg'] = 0.0
for i in range(len(D)):
    D['rating_avg'][i] = games[i].rating_average

D['pt'] = 0
for i in range(len(D)):
    D['pt'][i] = games[i].playing_time

any_in = lambda a, b: any(i in b for i in a)

germans = []
for i in range(len(D)):
    if any_in(categ[i], german):
        germans.append(i)

G = D.iloc[germans]

americans = []
for i in range(len(D)):
    if i not in germans:
        if any_in(categ[i], american):
            americans.append(i)

A = D.iloc[americans]


avg_num_voters = D['num_voters'].mean()
D2 = D[D['num_voters'] >= avg_num_voters]
D2 = D2.reset_index().drop(['index'], 1)

A2 = A[A['num_voters'] >= avg_num_voters]
A2 = A2.reset_index().drop(['index'], 1)

G2 = G[G['num_voters'] >= avg_num_voters]
G2 = G2.reset_index().drop(['index'], 1)

