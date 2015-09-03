import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn import tree

# First, with tags
songs_tags = pd.read_csv('https://raw.githubusercontent.com/wints/SF_DAT_15_WORK/master/HypeMachine_Final_Paper_Winters/songs_INDEXED_TAGLIST_NOECHO.csv')
songs_tags_dummies = songs_tags['tags'].str.get_dummies(sep=',')
songs_all_tags = songs_tags.merge(songs_tags_dummies, right_index=True, left_index=True)
songs_all_tags = songs_all_tags.drop(['artist', 'loved_count', 'posted_count', 'sitename', \
'time', 'title', 'tags', 'artist_index_mod', 'site_index_mod'], axis=1)
X, y = songs_all_tags.drop(['loved', 'mediaid'], axis = 1), songs_all_tags['loved']
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

ctree = tree.DecisionTreeClassifier()
ctree.fit(X_train, y_train)

features = X_train.columns.tolist()

ctree.classes_

# Which features are the most important?
ctree.feature_importances_

# Clean up the output
tags_importances = pd.DataFrame(zip(features, ctree.feature_importances_)).sort_index(by=1, ascending=False)

# next, going to get my tagsum column
tags_importances = tags_importances[tags_importances[1] > 0.005]
tags_list = tags_importances[0].tolist()

songs_all_tags_short = songs_all_tags[tags_list]
songs_all_tags_short['tagsum'] = songs_all_tags_short.sum(axis=1)
songs_tags_clean = songs_tags[['loved', 'sitename', 'mediaid', 'artist', 'title', 'time', \
'artist_index_mod', 'site_index_mod', 'loved_count', 'posted_count', 'tagsum']]




# Now with Echonest data
songs_echo = pd.read_csv('https://raw.githubusercontent.com/wints/SF_DAT_15_WORK/master/HypeMachine_Final_Paper_Winters/songs_ALL_ATTRIBUTES_PLUS_TAGSUM.csv')
songs_echo = songs_echo[['loved', 'valence', 'instrumentalness', 'loudness', 'speechiness'\
, 'tempo', 'time_signature', 'acousticness', 'danceability', 'duration', 'energy', 'liveness',\
'mediaid']]
songs_echo_short = songs_echo[songs_echo['valence'] > 0]
X, y = songs_echo_short.drop(['loved', 'mediaid'], axis = 1), songs_echo_short['loved']
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

ctree = tree.DecisionTreeClassifier(max_depth=100)
ctree.fit(X_train, y_train)

features = X_train.columns.tolist()

ctree.classes_

# Which features are the most important?
ctree.feature_importances_

# Clean up the output
echo_importances = pd.DataFrame(zip(features, ctree.feature_importances_)).sort_index\
(by=1, ascending=False)