import json
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.callbacks import CallbackAny2Vec
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


seq = []
for i in range(len(D)):
    seq.append(D['mechanics'][i])

all_mechs = list(itertools.chain(*seq))
m_freq = {m : all_mechs.count(m) for m in mechanics}

mfs = {k: v for k, v in sorted(m_freq.items(), key=lambda item: item[1], reverse=True)}

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
model = Word2Vec(seq, min_count=1, window=2, sg=1)
model.train(seq, total_examples=model.corpus_count, epochs=40, compute_loss=True, callbacks=[callback()])

# model = Word2Vec.load("data/mechanic2vec.model")

vocab = list(model.wv.vocab)

# CLUSTERING

num_clusters = 2
kmeans = cluster.KMeans(n_clusters=num_clusters)

X = model.wv.__getitem__(vocab)
pca = PCA(n_components=2)
px = pca.fit_transform(X)

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

#Cluster plot
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.scatter(df['x'], df['y'], c=clusters, s=30)

af = [[m, mfs[m]] for m in AL.index]
asm = np.array(sorted(af, key=lambda x: x[1]))
asm = asm[::-1]

gf = [[m, mfs[m]] for m in GL.index]
gsm = np.array(sorted(gf, key=lambda x: x[1]))
gsm = gsm[::-1]

for word, pos in df.iterrows():
    if word in asm[:, 0][:20]:
        ax.annotate(word, pos, color='black')
    elif word in gsm[:, 0][:20]:
        ax.annotate(word, pos, color='black')

for i, j in centroids:
    ax.scatter(i, j, s=50, c='red', marker='P')

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

AV = v_m[v_m['categ'] == 'american'].sort_values('dist')
GV = v_m[v_m['categ'] == 'german'].sort_values('dist')

#Games centers
m_dist = pd.DataFrame([], index=df.index, columns=['american_center', 'german_center'])

for mech in m_dist.index:
    dists = {categ: linalg_norm_T(df.loc[mech], game_centers[categ]) for categ in game_centers.__iter__()}
    m_dist.loc[mech] = dists['american_center'], dists['german_center']

AC = m_dist[m_dist['american_center'] < m_dist['german_center']].sort_values('american_center')
AC = AC.drop('german_center', axis=1)

GC = m_dist[m_dist['american_center'] > m_dist['german_center']].sort_values('german_center')
GC = GC.drop('american_center', axis=1)

print('AL ', len(AL))
print('GL ', len(GL))
print('AC ', len(AC))
print('GC ', len(GC))

colors = []
for mech in list(df.index):
    if mech in list(AC.index):
        colors.append('y')
    else:
        colors.append('purple')

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(df['x'], df['y'], c=colors, s=30)
for game in germans_title:
    ax.scatter(df.loc[gm[game]]['x'].mean(), df.loc[gm[game]]['y'].mean(), marker='X', color='blue')
    ax.annotate(game, (df.loc[gm[game]]['x'].mean(), df.loc[gm[game]]['y'].mean()))


for game in americans_title:
    ax.scatter(df.loc[am[game]]['x'].mean(), df.loc[am[game]]['y'].mean(), marker='*', color='orange')
    ax.annotate(game, (df.loc[am[game]]['x'].mean(), df.loc[am[game]]['y'].mean()))

ax.scatter(game_centers['american_center'][0], game_centers['american_center'][1], s=50, c='red', marker='P')
ax.scatter(game_centers['german_center'][0], game_centers['german_center'][1], s=50, c='red', marker='P')

o = pd.DataFrame([], index=df.index, columns=['AC', 'AL', 'GC', 'GL'])

for row in o.iterrows():
    o.loc[row[0]] = [m_dist.loc[row[0]]['american_center'], cluster_df2.loc[row[0]][ame], m_dist.loc[row[0]]['german_center'], cluster_df2.loc[row[0]][ger]]

o_max = max(o.max())
o_norm = o / o_max

y = [mech for mech in list(mfs.keys())]
y.reverse()

def submit(text):
    t = int(text)
    ax.clear()
    ax.axvline(x=0, color='black')
    ax.barh(y[-t:], np.array(1 - o_norm.loc[y[-t:]]['GL']))
    ax.barh(y[-t:], - np.array(1 - o_norm.loc[y[-t:]]['AL']), color='orange')

fig = plt.figure()
fig.suptitle('AMERICAN        GERMAN', x=0.54, y=0.94)
ax = fig.add_subplot(1, 1, 1)
plt.subplots_adjust(left=0.20, top=0.9)
plt.axvline(x=0, color='black')

ax.barh(y[-20:], np.array(1 - o_norm.loc[y[-20:]]['GL']))
ax.barh(y[-20:], - np.array(1 - o_norm.loc[y[-20:]]['AL']), color='orange')

initial_text = '20'
axbox = plt.axes([0.2, 0.01, 0.06, 0.03])
text_box = TextBox(axbox, 'Top mechanics', initial=initial_text)
text_box.on_submit(submit)

new_game = ['Market', 'Loans', 'Worker Placement']

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