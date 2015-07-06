import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

'''
I ran through this exercise with a partially-complete final project dataset. For the sake 
of simplicity, I only included numeric columns, all of which are relatively low-signal.
So while the predictions generated are garbage, it was still useful to walk through this process
(and confirm that, indeed, my hypothesis that predictions would be garbage was correct.)

The dataset in the read_csv below is also available in github.
'''
songs = pd.read_csv('songs_MASTER_DEDUPED.csv')

# dropping a couple garbage columns
songs = songs.drop('Unnamed: 0', axis=1)
songs = songs.drop('Unnamed: 0.1', axis=1)
# looks good now
songs.shape
songs.columns

# rudimentary normalization of loved_count, posted_count, time
loved_mean = songs.loved_count.mean()
loved_std = songs.loved_count.std() 
songs.loved_count = (songs.loved_count - loved_mean)/loved_std

posted_mean = songs.posted_count.mean()
posted_std = songs.posted_count.std() 
songs.posted_count = (songs.posted_count - posted_mean)/posted_std

time_mean = songs.time.mean()
time_std = songs.time.std()
songs.time = (songs.time - time_mean)/time_std

songs.head() # normalization looks ok

# create feature set and response set
X, y = songs.drop('loved', axis = 1), songs['loved']
X.shape
y.shape

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X, y)
knn.score(X, y)

''' 
Oops. I get 'ValueError: could not convert string to float: 
Carby (Feat. Ezra Koenig of Vampire Weekend)'. Not going to turn these string columns into dummies 
quite yet, so for simplicity's sake, I will drop non-numeric columns. So in the end, 
this probably won't generate a useful prediction because I expect the columns I'm keeping are 
low-signal by themselves, but it's good to go through the process anyway.
'''

songs = songs.drop('mediaid', axis=1)
songs = songs.drop('sitename', axis=1)
songs = songs.drop('title', axis=1)
songs = songs.drop('artist', axis=1)
X, y = songs.drop('loved', axis = 1), songs['loved']
X.shape
y.shape

# OK, let's try again
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X, y)
knn.score(X, y) # as expected with 1 neighbor, no train-test split: perfect score!

# now, time to start splitting data into train-test sets
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)

# the splits look good
X_train.shape
X_test.shape
y_train.shape
y_test.shape

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
knn.score(X_test, y_test) 
# as expected, using three numeric columns only, the pred is barely better than a random guess

# testing a couple other n_neighbors
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn.score(X_test, y_test)
# a wee bit better than 1 neighbor

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
knn.score(X_test, y_test)
# a wee bit worse than 1 neighbor

# and finally, time to cross-validate!
from sklearn.cross_validation import cross_val_score
knn = KNeighborsClassifier(n_neighbors=1)
scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
scores 
np.mean(scores) # ouch, pretty crappy results!

from sklearn.cross_validation import cross_val_score
knn = KNeighborsClassifier(n_neighbors=3)
scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
scores 
np.mean(scores) # worst of the three

from sklearn.cross_validation import cross_val_score
knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
scores 
np.mean(scores) # BARELY better than randomly guessing

# finding an optimal k
k_range = range(1, 30, 2)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores.append(np.mean(cross_val_score(knn, X, y, cv=5, scoring='accuracy')))
    # confirms that k = 5 is best value
scores

# plotting different values of k
plt.figure()
plt.plot(k_range, scores) 

# playing around with GridSearch
from sklearn.grid_search import GridSearchCV
knn = KNeighborsClassifier()
k_range = range(1, 30, 2)
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





