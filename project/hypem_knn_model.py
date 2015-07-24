# knn model 

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns

songs_all = pd.read_csv('/Users/wints/Desktop/ga_data_science/SF_DAT_15/project/songs_MASTER_INDEXED.csv') # creates songs_all dataframe


# normalize loved_count and posted_count
loved_mean = songs_all.loved_count.mean()
loved_std = songs_all.loved_count.std() 
songs_all.loved_count = (songs_all.loved_count - loved_mean)/loved_std
posted_mean = songs_all.posted_count.mean()
posted_std = songs_all.posted_count.std() 
songs_all.posted_count = (songs_all.posted_count - posted_mean)/posted_std

X, y = songs_all.drop(['loved', 'time' , 'mediaid', 'title', 'sitename', 'artist'], axis = 1), songs_all['loved']
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

knn.fit(X_train, y_train)

from sklearn.grid_search import GridSearchCV
knn = KNeighborsClassifier()
k_range = range(1, 30)
param_grid = dict(n_neighbors=k_range)
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
grid.fit(X, y)

grid.grid_scores_
grid_mean_scores = [result[1] for result in grid.grid_scores_]

grid.best_score_     # shows best score
grid.best_params_    # shows params that generate best score
grid.best_estimator_ # entire model that generates best score

knn_top = KNeighborsClassifier(n_neighbors=28)
knn_top.fit(X_train, y_train)
y_pred_class = knn.predict(X_test)

from sklearn import metrics
print metrics.accuracy_score(y_test, y_pred_class)

print metrics.confusion_matrix(y_test, y_pred_class)
cm = metrics.confusion_matrix(y_test, y_pred_class)

specificity = (float(cm[0,0]) / (cm[0,0] + cm[0,1])) 
sensitivity = (float(cm[1,1]) / (cm[1,1] + cm[1,0])) 