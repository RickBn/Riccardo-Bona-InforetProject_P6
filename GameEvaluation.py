import json
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from sklearn import cluster
from sklearn import metrics
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
import itertools

D = pd.read_json("data/ggt3.json")
titles = D['title']
mechs = D['mechanics']

mechanics = list()
for i in range(len(D)):
    for m in D['mechanics'][i]:
        if m not in mechanics:
            mechanics.append(m)

# Mechanic2Vec

seq = []
for i in range(len(D)):
    seq.append(D['mechanics'][i])

model = {}
model = Word2Vec(seq, min_count=1, window=2)

model.train(seq, total_examples=model.corpus_count, epochs=40)

vocab = list(model.wv.vocab)
X = model.wv.__getitem__(vocab)

# CLUSTERING

pca = PCA(n_components=2)
px = pca.fit_transform(X)

num_clusters = 2
kmeans = cluster.KMeans(n_clusters=num_clusters)

kmeans.fit(px)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_

clusters = np.array(labels)

#Vocab dataframe

df = pd.DataFrame(px, index=vocab, columns=['x', 'y'])

#2 CLUSTERS
cluster_df = pd.DataFrame([], index=vocab, columns=[0, 1])
cluster_df2 = pd.DataFrame([], index=vocab, columns=[0, 1])

for i, words in enumerate(df.iterrows()):
    cluster = labels[i]
    centroid = centroids[cluster]
    cluster_df.loc[words[0]][cluster] = np.linalg.norm(np.array(words[1]) - centroid)

for i, words in enumerate(df.iterrows()):
    for cluster in labels:
        centroid = centroids[cluster]
        cluster_df2.loc[words[0]][cluster] = np.linalg.norm(np.array(words[1]) - centroid)

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

o = pd.DataFrame([], index=df.index, columns=['AL', 'GL'])

for row in o.iterrows():
    o.loc[row[0]] = [cluster_df2.loc[row[0]][ame], cluster_df2.loc[row[0]][ger]]

o_max = max(o.max())

o_norm = o / o_max

all_mechs = list(itertools.chain(*seq))
m_freq = {m : all_mechs.count(m) for m in o_norm.index}

mfs = {k: v for k, v in sorted(m_freq.items(), key=lambda item: item[1], reverse=True)}

y = [mech for mech in list(mfs.keys())]
y.reverse()

fig = plt.figure()
fig.suptitle('AMERICAN-LIKE    GERMAN-LIKE', x=0.54, y=0.94)
ax = fig.add_subplot(1, 1, 1)
plt.subplots_adjust(left=0.20, top=0.9)
plt.axvline(x=0, color='black')

ax.barh(y[-20:], np.array(1 - o_norm.loc[y[-20:]]['GL']))
ax.barh(y[-20:], - np.array(1 - o_norm.loc[y[-20:]]['AL']), color='orange')

initial_text = '20'
axbox = plt.axes([0.2, 0.01, 0.06, 0.03])
text_box = TextBox(axbox, 'Top mechanics', initial=initial_text)

def submit(text):
    t = int(text)
    ax.clear()
    ax.axvline(x=0, color='black')
    ax.barh(y[-t:], np.array(1 - o_norm.loc[y[-t:]]['GL']))
    ax.barh(y[-t:], - np.array(1 - o_norm.loc[y[-t:]]['AL']), color='orange')

text_box.on_submit(submit)


new_game = ['Market', 'Loans', 'Acting']

def plot_custom_game(ngo, ng, ngv):
    fig = plt.figure()
    fig.suptitle('CUSTOM GAME')
    plt.subplots_adjust(left=0.20, top=0.9)

    ax1 = plt.subplot2grid((3,3), (0,0), rowspan=3, colspan=2)
    ax1.set_title('American                           German')
    ax1.axvline(x=0, color='black')
    ax1.barh(list(ng.keys()), np.array(1 - ngv[:, 1]), color='blue')
    ax1.barh(list(ng.keys()), - np.array(1 - ngv[:, 0]), color='orange')

    ax2 = plt.subplot2grid((3,3), (1,2) , rowspan=3, colspan=2)
    ax2.bar(['American', 'German'], (1 - np.array(ngo)), color=['orange', 'blue'])

def evaluate_game(mechanics):
    nm = [[m, mfs[m]] for m in mechanics]
    nms = np.array(sorted(nm, key=lambda x: x[1]))

    ng = {}
    for m in nms[:, 0]:
        ng[m] = [o_norm.loc[m]['AL'], o_norm.loc[m]['GL']]

    ngv = np.array(list(ng.values()))
    ngo = [np.mean(ngv[:, 0]), np.mean(ngv[:, 1])]

    plot_custom_game(ngo, ng, ngv)

    print('American: ', 1 - ngo[0], '\n' + 'German: ', 1 - ngo[1], '\n')

    for m in reversed(nms[:, 0]):
        print(m + ": ", '[American: ', 1 - ng[m][0], '],', '[German: ', 1 - ng[m][1], ']')

evaluate_game(new_game)

def game_in_space(mechanics):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(df['x'], df['y'], c=clusters, s=30)

    for m in mechanics:
        if  m in AL.index:
            ax.annotate(m, df.loc[m], color='orange')
        elif m in GL.index:
            ax.annotate(m, df.loc[m], color='blue')

    for i, j in centroids:
        ax.scatter(i, j, s=50, c='red', marker='+')

    ax.scatter(df.loc[mechanics]['x'].mean(), df.loc[mechanics]['y'].mean(), marker='*', s=50, c='m')
    ax.annotate('GAME', (df.loc[mechanics]['x'].mean(), df.loc[mechanics]['y'].mean()), c='m')

game_in_space(new_game)