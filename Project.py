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

#Test
germans = ['catan', 'puerto rico', 'carcassonne', 'ticket ride', 'alhambra']
G = D[D['title'].isin(germans)]
G = G.reset_index().drop(['index'], 1)

gm = {}
for i in range(len(germans)):
    gm[germans[i]] = G['mechanics'][i]

g_mec = list(gm.values())

americans = ['axis & allies', 'dune', 'cosmic encounter', 'talisman', 'twilight imperium']
A = D[D['title'].isin(americans)]
A = A.reset_index().drop(['index'], 1)
A = A.drop([2, 4, 6, 8])
A = A.reset_index().drop(['index'], 1)

am = {}
for i in range(len(americans)):
    am[americans[i]] = A['mechanics'][i]

a_mec = list(am.values())

g = []
for i in g_mec:
    for j in i:
        if j not in g:
            g.append(j)

a = []
for i in a_mec:
    for j in i:
        if j not in a:
            a.append(j)

#Mechanic2Vec

from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
import seaborn as sns
import matplotlib.pyplot as plt

seq = []
for i in range(len(D)):
    seq.append(D['mechanics'][i])

model = {}
model = Word2Vec(seq, min_count=1, window=2)

model.train(seq, total_examples=model.corpus_count, epochs=80)

vocab = list(model.wv.vocab)
X = model.wv.__getitem__(vocab)

sim = []
for m1 in vocab:
    m_sim = []
    for m2 in vocab:
        m_sim.append(model.wv.similarity(m1, m2))
    sim.append(m_sim)

sim_df = df = pd.DataFrame(sim, index=vocab, columns=vocab)



# mech = 'Action Drafting'
# a = dict(model.wv.most_similar(positive=mech))
# a_k = a.keys()
# a_v = a.values()

#CLUSTERING

#SKLEARN
from sklearn import cluster
from sklearn import metrics

num_clusters = 3

kmeans = cluster.KMeans(n_clusters=num_clusters)
kmeans.fit(sim)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_

clusters = np.array(labels)

# wcss = []
# for i in range(1, 11):
#     kmeans = cluster.KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
#     kmeans.fit(sim)
#     wcss.append(kmeans.inertia_)
# plt.plot(range(1, 11), wcss)
# plt.title('Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
# plt.show()

print("Cluster id labels for inputted data")
print(labels)
print("Centroids data")
print(centroids)

print("Score (Opposite of the value of X on the K-means objective which is Sum of distances of samples to their closest cluster center):")
#print(kmeans.score(X))
print(kmeans.score(sim))


#silhouette_score = metrics.silhouette_score(X, labels, metric='euclidean')
silhouette_score = metrics.silhouette_score(sim, labels, metric='euclidean')

print("Silhouette_score: ")
print(silhouette_score)

#PLOT

from sklearn.decomposition import PCA
pca = PCA(n_components=2)

ps = pca.fit_transform(sim)
cs = pca.fit_transform(centroids)

df = pd.DataFrame(ps, index=vocab, columns=['x', 'y'])

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.scatter(df['x'], df['y'], c=clusters, s=30)

# for word, pos in df.iterrows():
#         ax.annotate(word, pos)

for word, pos in df.iterrows():
    if word in a and word in g:
        ax.annotate(word, pos, color='blue')
    elif word in a:
        ax.annotate(word, pos, color='red')
    elif word in g:
        ax.annotate(word, pos, color='green')

for i, j in cs:
    ax.scatter(i, j, s=50, c='red', marker='+')

