import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

tags_in = pd.read_csv('/Users/wints/Desktop/ga_data_science/SF_DAT_15/project/tags_list_MASTER_DUMMIES_CONCAT.csv') # creates songs_all dataframe
songs_all = pd.read_csv('/Users/wints/Desktop/ga_data_science/SF_DAT_15/project/songs_MASTER_DEDUPED_csv.csv') # creates songs_all dataframe

tags_in['tags'] = tags_in['tags'].str.replace("u'","'")
tags_in['tags'] = tags_in['tags'].str.replace("[","")
tags_in['tags'] = tags_in['tags'].str.replace("]","")

tags_in = tags_in[['mediaid', 'tags']]
songs_lim = songs_all[['mediaid', 'loved']]
tags_in.sort(columns='mediaid', inplace=True)
tags_in.reset_index(drop=True, inplace=True)
songs_lim.sort(columns='mediaid', inplace=True)
songs_lim.reset_index(drop=True, inplace=True)
comb = [tags_in, songs_lim]
tags_loved = pd.concat(comb, axis=1)
tags_loved.columns = ['mediaid', 'tags', 'placeholder', 'loved']
tags_loved = tags_loved[['mediaid', 'tags', 'loved']]


tags_true = tags_loved[tags_loved.loved == True]
tags_false = tags_loved[tags_loved.loved == False]

tags_true_dummies = tags_true['tags'].str.get_dummies(sep=', ')
tags_true = tags_true.merge(tags_true_dummies, right_index=True, left_index=True)
tags_true.drop(['tags', 'loved'], inplace=True, axis=1)

tags_false_dummies = tags_false['tags'].str.get_dummies(sep=', ')
tags_false = tags_false.merge(tags_false_dummies, right_index=True, left_index=True)
tags_false.drop(['tags', 'loved'], inplace=True, axis=1)

# clustering for tags_true
song_cluster = KMeans(n_clusters=4, init='random')
song_cluster.fit(tags_true.drop('mediaid', axis=1))
y_kmeans = song_cluster.predict(tags_true.drop('mediaid', axis=1))

for cluster_in_question in range(0,4):
    # get center of cluster
    "centroid", song_cluster.cluster_centers_[cluster_in_question]
    # grab songs in dataframe that belong to this cluster
    songs = tags_true[np.where(y_kmeans == cluster_in_question, True, False)]['mediaid']
    # look at top five qualities in cluster
    print sorted(zip(tags_true.columns[1:], song_cluster.cluster_centers_[cluster_in_question]), key=lambda x:x[1], reverse=True)[1:6]
    print
    

k_rng = range(1,15)
est = [KMeans(n_clusters = k).fit(tags_true.drop('mediaid',axis=1)) for k in k_rng]

# calculate silhouette score
from sklearn import metrics
silhouette_score = [metrics.silhouette_score(tags_true.drop('mediaid',axis=1), e.labels_, metric='euclidean') for e in est[1:]]

# Plot the results
plt.figure(figsize=(7, 8))
plt.subplot(211)
plt.title('loved=true k choice')
plt.plot(k_rng[1:], silhouette_score, 'b*-')
plt.xlim([1,15])
plt.grid(True)
plt.ylabel('Silhouette Coefficient')
plt.plot(4,silhouette_score[2], 'o', markersize=12, markeredgewidth=1.5,
markerfacecolor='None', markeredgecolor='r')

# clustering for tags_false
song_cluster = KMeans(n_clusters=4, init='random')
song_cluster.fit(tags_false.drop('mediaid', axis=1))
y_kmeans = song_cluster.predict(tags_false.drop('mediaid', axis=1))

for cluster_in_question in range(0,4):
    # get center of cluster
    "centroid", song_cluster.cluster_centers_[cluster_in_question]
    # grab songs in dataframe that belong to this cluster
    songs = tags_false[np.where(y_kmeans == cluster_in_question, True, False)]['mediaid']
    # look at top five qualities in cluster
    print sorted(zip(tags_false.columns[1:], song_cluster.cluster_centers_[cluster_in_question]), key=lambda x:x[1], reverse=True)[1:6]
    print

k_rng = range(1,15)
est = [KMeans(n_clusters = k).fit(tags_false.drop('mediaid',axis=1)) for k in k_rng]

# calculate silhouette score
from sklearn import metrics
silhouette_score = [metrics.silhouette_score(tags_false.drop('mediaid',axis=1), e.labels_, metric='euclidean') for e in est[1:]]

# Plot the results
plt.figure(figsize=(7, 8))
plt.subplot(211)
plt.title('Using the elbow method to inform k choice')
plt.plot(k_rng[1:], silhouette_score, 'b*-')
plt.xlim([1,15])
plt.grid(True)
plt.ylabel('Silhouette Coefficient')
plt.plot(4,silhouette_score[2], 'o', markersize=12, markeredgewidth=1.5,
markerfacecolor='None', markeredgecolor='r')