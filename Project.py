import json
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from string import punctuation
from nltk.corpus import stopwords
import spacy
import nltk
import numpy as np

D = pd.read_json("data/ggt3.json")
titles = D['title']
desc = D['description']
categ = D['categories']
mechs = D['mechanics']

# import pickle
#
# with open('data\games.json', 'rb') as fp:
#     games = pickle.load(fp)

mechanics = list()
for i in range(len(D)):
    for m in D['mechanics'][i]:
        if m not in mechanics:
            mechanics.append(m)

from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

#Mechanic2Vec

seq = []
for i in range(len(D)):
    seq.append(D['mechanics'][i])

model = {}
model = Word2Vec(seq, min_count=1)

model.train(seq, total_examples=model.corpus_count, epochs=50)

#Test

mech = 'Action Drafting'
a = dict(model.wv.most_similar(positive=mech))
a_k = a.keys()
a_v = a.values()

#PLOT
vocab = list(model.wv.vocab)
X = model[vocab]
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)

df = pd.DataFrame(X_tsne, index=vocab, columns=['x', 'y'])

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.scatter(df['x'], df['y'])

for word, pos in df.iterrows():
    if word == mech:
        ax.annotate(word, pos, color='red')
    elif word in list(a_k):
        ax.annotate(word, pos, color='orange')
    else:
        ax.annotate(word, pos)
