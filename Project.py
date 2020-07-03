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

# Test
germans = [13, 3076, 822, 9209, 6249, 120677, 84876, 42, 18602, 2651]
G = D[D['id'].isin(germans)]
G = G.reset_index().drop(['index'], 1)
germans_title = list(G['title'])

gm = {}
for i in range(len(germans_title)):
    gm[germans_title[i]] = G['mechanics'][i]

g_mec = list(gm.values())

americans = [98, 121, 39463, 714, 233078, 113924, 174430, 15987, 37111, 12350]
A = D[D['id'].isin(americans)]
A = A.reset_index().drop(['index'], 1)
americans_title = list(A['title'])

am = {}
for i in range(len(americans_title)):
    am[americans_title[i]] = A['mechanics'][i]

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

# Mechanic2Vec

from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.callbacks import CallbackAny2Vec
import seaborn as sns
import matplotlib.pyplot as plt

seq = []
for i in range(len(D)):
    seq.append(D['mechanics'][i])


class callback(CallbackAny2Vec):

    # Callback to print loss after each epoch
    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        if self.epoch == 0:
            print('Loss after epoch {}: {}'.format(self.epoch, loss))
        else:
            print('Loss after epoch {}: {}'.format(self.epoch, loss - self.loss_previous_step))
        self.epoch += 1
        self.loss_previous_step = loss


model = {}
model = Word2Vec(seq, min_count=1, window=2)

model.train(seq, total_examples=model.corpus_count, epochs=40, compute_loss=True, callbacks=[callback()])

vocab = list(model.wv.vocab)
X = model.wv.__getitem__(vocab)

sim = []
for m1 in vocab:
    m_sim = []
    for m2 in vocab:
        m_sim.append(model.wv.similarity(m1, m2))
    sim.append(m_sim)

sim_df = df = pd.DataFrame(sim, index=vocab, columns=vocab)

# CLUSTERING

# SKLEARN
from sklearn import cluster
from sklearn import metrics
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
px = pca.fit_transform(X)

num_clusters = 2
kmeans = cluster.KMeans(n_clusters=num_clusters)

kmeans.fit(px)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_

clusters = np.array(labels)

# wcss = []
# for i in range(1, 11):
#     kmeans = cluster.KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
#     kmeans.fit(px)
#     wcss.append(kmeans.inertia_)
# plt.plot(range(1, 11), wcss)
# plt.title('Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
# plt.show()

print(labels)

print(centroids)

print(kmeans.score(px))

silhouette_score = metrics.silhouette_score(px, labels, metric='euclidean')

print(silhouette_score)

# PLOT

df = pd.DataFrame(px, index=vocab, columns=['x', 'y'])

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.scatter(df['x'], df['y'], c=clusters, s=30)

# for word, pos in df.iterrows():
#         ax.annotate(word, pos)

# for word, pos in df.iterrows():
#     if word in a and word in g:
#         ax.annotate(word, pos, color='blue')
#     elif word in a:
#         ax.annotate(word, pos, color='red')
#     elif word in g:
#         ax.annotate(word, pos, color='green')

for i, j in centroids:
    ax.scatter(i, j, s=50, c='red', marker='+')


#2 CLUSTERS
cluster_df = pd.DataFrame([], index=vocab, columns=[0, 1])

for i, words in enumerate(df.iterrows()):
    cluster = labels[i]
    centroid = centroids[cluster]
    cluster_df.loc[words[0]][cluster] = np.linalg.norm(np.array(words[1]) - centroid)

if np.isnan(cluster_df.loc['Campaign / Battle Card Driven'][0]) == False:
    ame = 0
    ger = 1
else:
    ame = 1
    ger = 0


AL = cluster_df[cluster_df[ame].isna() == False]
AL = AL.drop(ger, axis=1)
AL.columns = ['dist']
AL = AL.sort_values('dist')

GL = cluster_df[cluster_df[ger].isna() == False]
GL = GL.drop(ame, axis=1)
GL.columns = ['dist']
GL = GL.sort_values('dist')

a_dists = AL.loc[:]['dist'].values
a_d_norm = a_dists / max(a_dists)
AL.insert(1, 'dist_norm', a_d_norm, True)

g_dists = GL.loc[:]['dist'].values
g_d_norm = g_dists / max(g_dists)
GL.insert(1, 'dist_norm', g_d_norm, True)

#3 CLUSTERS
# num_clusters = 3
# kmeans = cluster.KMeans(n_clusters=num_clusters)
#
# kmeans.fit(px)
#
# labels = kmeans.labels_
# centroids = kmeans.cluster_centers_
#
# clusters = np.array(labels)
#
# # wcss = []
# # for i in range(1, 11):
# #     kmeans = cluster.KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
# #     kmeans.fit(px)
# #     wcss.append(kmeans.inertia_)
# # plt.plot(range(1, 11), wcss)
# # plt.title('Elbow Method')
# # plt.xlabel('Number of clusters')
# # plt.ylabel('WCSS')
# # plt.show()
#
# print(labels)
#
# print(centroids)
#
# print(kmeans.score(px))
#
# silhouette_score = metrics.silhouette_score(px, labels, metric='euclidean')
#
# print(silhouette_score)
#
# # PLOT
#
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
#
# ax.scatter(df['x'], df['y'], c=clusters, s=30)
#
# # for word, pos in df.iterrows():
# #         ax.annotate(word, pos)
#
# for word, pos in df.iterrows():
#     if word in a and word in g:
#         ax.annotate(word, pos, color='blue')
#     elif word in a:
#         ax.annotate(word, pos, color='red')
#     elif word in g:
#         ax.annotate(word, pos, color='green')
#
# for i, j in centroids:
#     ax.scatter(i, j, s=50, c='red', marker='+')
#
# cluster_df = pd.DataFrame([], index=vocab, columns=[0, 1, 2])
#
# for i, words in enumerate(df.iterrows()):
#     cluster = labels[i]
#     centroid = centroids[cluster]
#     cluster_df.loc[words[0]][cluster] = np.linalg.norm(np.array(words[1]) - centroid)
#
# AL = cluster_df[cluster_df[2].isna() == False]
# AL = AL.drop([0,1], axis=1)
# AL.columns = ['americanlike']
# AL = AL.sort_values('americanlike')
#
# GL = cluster_df[cluster_df[0].isna() == False]
# GL = GL.drop([1,2], axis=1)
# GL.columns = ['germanlike']
# GL = GL.sort_values('germanlike')



#NN

v_a = pd.DataFrame([], index=americans_title, columns=['x', 'y'])
v_g = pd.DataFrame([], index=germans_title, columns=['x', 'y'])

for game in americans_title:
    v_a.loc[game] = df.loc[am[game]]['x'].mean(), df.loc[am[game]]['y'].mean()

for game in germans_title:
    v_g.loc[game] = df.loc[gm[game]]['x'].mean(), df.loc[gm[game]]['y'].mean()

def linalg_norm_T(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

games_df = pd.concat([v_a, v_g])
v_m = pd.DataFrame([], index=df.index, columns=['game', 'dist', 'categ'])
for mech in df.index:
    dists = {game[0]: linalg_norm_T(df.loc[mech], game[1]) for game in games_df.iterrows()}
    min_g = min(dists, key=dists.get)
    min_d = dists[min_g]
    categ = ''
    if min_g in americans_title:
        categ = 'american'
    else:
        categ = 'german'
    v_m.loc[mech] = [min_g, min_d, categ]

game_centers = {'american_center' : [v_a['x'].mean(), v_a['y'].mean()], 'german_center' : [v_g['x'].mean(), v_g['y'].mean()]}

colors = np.where(v_m['categ'] == 'american', 'y', 'm')
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(df['x'], df['y'], c=colors, s=30)

for game in germans_title:
    ax.scatter(df.loc[gm[game]]['x'].mean(), df.loc[gm[game]]['y'].mean(), marker='X')
    ax.annotate(game, (df.loc[gm[game]]['x'].mean(), df.loc[gm[game]]['y'].mean()))


for game in americans_title:
    ax.scatter(df.loc[am[game]]['x'].mean(), df.loc[am[game]]['y'].mean(), marker='*')
    ax.annotate(game, (df.loc[am[game]]['x'].mean(), df.loc[am[game]]['y'].mean()))


ax.scatter(game_centers['american_center'][0], game_centers['american_center'][1], s=50, c='red', marker='P')
ax.scatter(game_centers['german_center'][0], game_centers['german_center'][1], s=50, c='red', marker='P')

AV = v_m[v_m['categ'] == 'american'].sort_values('dist')
GV = v_m[v_m['categ'] == 'german'].sort_values('dist')

a_dists = AV.loc[:]['dist'].values
a_d_norm = a_dists / max(a_dists)
AV.insert(3, 'dist_norm', a_d_norm, True)

g_dists = GV.loc[:]['dist'].values
g_d_norm = g_dists / max(g_dists)
GV.insert(3, 'dist_norm', g_d_norm, True)


#Games centers
m_dist = pd.DataFrame([], index=df.index, columns=['american_center', 'german_center'])

for mech in m_dist.index:
    dists = {categ: linalg_norm_T(df.loc[mech], game_centers[categ]) for categ in game_centers.__iter__()}
    m_dist.loc[mech] = dists['american_center'], dists['german_center']

AC = m_dist[m_dist['american_center'] < m_dist['german_center']].sort_values('american_center')
AC = AC.drop('german_center', axis=1)

GC = m_dist[m_dist['american_center'] > m_dist['german_center']].sort_values('german_center')
GC = GC.drop('american_center', axis=1)

ac_dists = AC.loc[:]['american_center'].values
ac_d_norm = ac_dists / max(ac_dists)
AC.insert(1, 'dist_norm', ac_d_norm, True)

gc_dists = GC.loc[:]['german_center'].values
gc_d_norm = gc_dists / max(gc_dists)
GC.insert(1, 'dist_norm', gc_d_norm, True)

colors = []
for mech in list(df.index):
    if mech in list(AC.index):
        colors.append('y')
    else:
        colors.append('m')

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(df['x'], df['y'], c=colors, s=30)
ax.scatter(game_centers['american_center'][0], game_centers['american_center'][1], s=50, c='red', marker='P')
ax.scatter(game_centers['german_center'][0], game_centers['german_center'][1], s=50, c='red', marker='P')


fig, axs = plt.subplots(3)
fig.suptitle('AMERICAN-LIKE MECHANICS')

axs[0].set_title('Distance from games center')
axs[0].barh(AC.index[:10], 1 - AC['dist_norm'][:10])
axs[0].invert_yaxis()

axs[1].set_title('Distance from cluster centroid')
axs[1].barh(AL.index[:10], 1 - AL['dist_norm'][:10], color='orange')
axs[1].invert_yaxis()

axs[2].set_title('Distance from nearest game')
axs[2].barh(AV.index[:10], 1 - AV['dist_norm'][:10], color='green')
axs[2].invert_yaxis()


fig, axs = plt.subplots(3)
fig.suptitle('GERMAN-LIKE MECHANICS')

axs[0].set_title('Distance from games center')
axs[0].barh(GC.index[:10], 1 - GC['dist_norm'][:10])
axs[0].invert_yaxis()

axs[1].set_title('Distance from cluster centroid')
axs[1].barh(GL.index[:10], 1 - GL['dist_norm'][:10], color='orange')
axs[1].invert_yaxis()

axs[2].set_title('Distance from nearest game')
axs[2].barh(GV.index[:10], 1 - GV['dist_norm'][:10], color='green')
axs[2].invert_yaxis()

#Wargames

wargames = [12333, 9203, 103885, 227460, 132018, 128996]
W = D[D['id'].isin(wargames)]
W = W.reset_index().drop(['index'], 1)
wargames_title = list(W['title'])

wm = {}
for i in range(len(wargames_title)):
    wm[wargames_title[i]] = W['mechanics'][i]

w_mec = list(wm.values())

v_w = pd.DataFrame([], index=wargames_title, columns=['x', 'y'])

for game in wargames_title:
    v_w.loc[game] = df.loc[wm[game]]['x'].mean(), df.loc[wm[game]]['y'].mean()

for game in wargames_title:
    ax.scatter(v_w.loc[game]['x'], v_w.loc[game]['y'], marker='D')
    ax.annotate(game, (v_w.loc[game]['x'], v_w.loc[game]['y']))