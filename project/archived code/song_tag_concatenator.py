import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

songs_all = pd.read_csv('/Users/wints/Desktop/ga_data_science/SF_DAT_15/project/songs_MASTER_DEDUPED_csv.csv') # creates songs_all dataframe
tags_all = pd.read_csv('/Users/wints/Desktop/ga_data_science/SF_DAT_15/tags_list_MASTER_DUMMIES_CONCAT.csv') # creates songs_all dataframe
tags_all.reset_index(drop=True, inplace=True)
tags_all.head()
songs_all.head()

# create a master data file where nothing has been  made into a dummy
master_concat = (songs_all, tags_all)
songs = pd.concat(master_concat, axis=1)
songs.to_csv('song_list_MASTER_TAGDUMMIES_2.csv', index=False)

# create an artist dummies frame 
artist_dummies = master_file_tagdummies['artist'].str.get_dummies()
dummies_concat = (artist_dummies, master_file_tagdummies)

# create a frame with both artist and tag dummied
master_file_tag_artist_dummies = pd.concat(dummies_concat, axis=1)
master_file_tag_artist_dummies = pd.concat(dummies_concat, axis=1)
master_file_tag_artist_dummies.to_csv('song_list_MASTER_TAG_ARTIST_DUMMIES.csv', index=False)

# for easy regerence
songs = pd.concat(dummies_concat, axis=1)

# rudimentary normalization of loved_count, posted_count, time
loved_mean = songs.loved_count.mean()
loved_std = songs.loved_count.std() 
songs.loved_count = (songs.loved_count - loved_mean)/loved_std

posted_mean = songs.posted_count.mean()
posted_std = songs.posted_count.std() 
songs.posted_count = (songs.posted_count - posted_mean)/posted_std

X, y = songs.drop(['loved', 'time', 'mediaid', 'tags', 'title', 'sitename', 'artist'], axis = 1), songs['loved']
X.shape
y.shape
X.columns

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X, y)
knn.score(X, y) 


from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# the splits look good
X_train.shape
X_test.shape
y_train.shape
y_test.shape

knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)
knn.score(X_test, y_test) 

# playing around with GridSearch
from sklearn.grid_search import GridSearchCV
knn = KNeighborsClassifier()
k_range = range(1, 30)
param_grid = dict(n_neighbors=k_range) # key has to be exactly the name as scikit learn calls it
grid = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid.fit(X, y)

grid.grid_scores_
grid_mean_scores = [result[1] for result in grid.grid_scores_]

# plot the GridSearch results
plt.figure()
plt.plot(k_range, grid_mean_scores)

# look into what GridSearch reveals
grid.best_score_     # shows us the best score
grid.best_params_    # shows us the optimal parameters
grid.best_estimator_ # details of the best-fit model

from sklearn.cross_validation import cross_val_score
knn = KNeighborsClassifier(n_neighbors=2)
scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
scores 
np.mean(scores) # knn doesn't necessarily get me a better score

y_all_no = [0] * len(y_test)
y_all_yes = [1] * len(y_test)
np.mean(y_all_yes == y_test) # 51.5%
np.mean(y_all_no == y_test) # 48.4%%

# testing logistic regression instead
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred_class = logreg.predict(X_test)
from sklearn import metrics
print metrics.accuracy_score(y_test, y_pred_class)
np.mean(y_pred_class == y_test) # Hooray! 58.59%, which is better than guessing 'all yes' or 'all no'


prds = logreg.predict(X)
print metrics.confusion_matrix(y_test, y_pred_class)
