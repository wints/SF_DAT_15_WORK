import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn import svm, linear_model, datasets
from sklearn.pipeline import Pipeline

# update to use github URL
songs = pd.read_csv('songs_FINAL_DOUBLE_TAGSUM.csv')
songs = songs[songs['speechiness'] > 0]

# define features and response
X, y = songs.drop(['loved', 'artist', 'title', 'sitename', 'mediaid', \
'tagsum'], axis = 1), songs['loved']
X_train, X_test, y_train, y_test = train_test_split(X, y)

# cross-validate for logistic regression
logreg = LogisticRegression()
cross_val_score(logreg, X, y, scoring='accuracy', cv=10).mean()
cross_val_score(logreg, X, y, scoring='roc_auc', cv=10).mean() 

# generate preds
logreg.fit(X_train, y_train)
y_pred_class = logreg.predict(X_test)

# logistic regression confusion matrix and specificity / sensitivity
print metrics.confusion_matrix(y_test, y_pred_class)
cm = metrics.confusion_matrix(y_test, y_pred_class)
specificity = (float(cm[0,0]) / (cm[0,0] + cm[0,1])) 
print specificity
sensitivity = (float(cm[1,1]) / (cm[1,1] + cm[1,0])) 
print sensitivity

# grid search / cross-validate for decision tree
ctree = tree.DecisionTreeClassifier()
depth_range = range(1, 20)
criterion_range = ['gini', 'entropy']
max_feaure_range = range(1,7)
param_grid = dict(max_depth=depth_range, criterion=criterion_range, \
max_features=max_feaure_range)
grid = GridSearchCV(ctree, param_grid, cv=5, scoring='accuracy')
grid.fit(X, y)
print grid.best_score_

depth_range = range(1, 20)
criterion_range = ['gini', 'entropy']
max_feaure_range = range(1,6)
param_grid = dict(max_depth=depth_range, criterion=criterion_range, \
max_features=max_feaure_range)
grid = GridSearchCV(ctree, param_grid, cv=5, scoring='roc_auc')
grid.fit(X, y)
print grid.best_score_

# generate preds with best tree params
best = grid.best_estimator_
best.fit(X_train, y_train)
y_pred_class = best.predict(X_test)

# tree confusion matrix and specificity / sensitivity
print metrics.confusion_matrix(y_test, y_pred_class)
cm = metrics.confusion_matrix(y_test, y_pred_class)
specificity = (float(cm[0,0]) / (cm[0,0] + cm[0,1])) 
print specificity
sensitivity = (float(cm[1,1]) / (cm[1,1] + cm[1,0])) 
print sensitivity

# grid search / cross-validate for knn
knn = KNeighborsClassifier()
k_range = range(1, 30)
param_grid = dict(n_neighbors=k_range)
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
grid.fit(X, y)
print grid.best_score_

knn = KNeighborsClassifier()
k_range = range(1, 30)
param_grid = dict(n_neighbors=k_range)
grid = GridSearchCV(knn, param_grid, cv=10, scoring='roc_auc')
grid.fit(X, y)
print grid.best_score_
grid.best_params_ # best params for maximizing roc_auc

# generate preds with best knn params
knn_top = KNeighborsClassifier(n_neighbors=21)
knn_top.fit(X_train, y_train)
y_pred_class = knn_top.predict(X_test)

# knn confusion matrix and specificity / sensitivity
print metrics.confusion_matrix(y_test, y_pred_class)
cm = metrics.confusion_matrix(y_test, y_pred_class)
specificity = (float(cm[0,0]) / (cm[0,0] + cm[0,1])) 
print specificity
sensitivity = (float(cm[1,1]) / (cm[1,1] + cm[1,0])) 
print sensitivity

# cross-validate for naive bayes
nb = MultinomialNB()
from sklearn.cross_validation import cross_val_score
cross_val_score(nb, X, y, cv=10, scoring='accuracy').mean() 
cross_val_score(nb, X, y, cv=10, scoring='roc_auc').mean() 

# generate preds for nb
nb.fit(X_train, y_train)
y_pred_class = nb.predict(X_test)

# nb confusion matrix and specificity / sensitivity
print metrics.confusion_matrix(y_test, y_pred_class)
cm = metrics.confusion_matrix(y_test, y_pred_class)
specificity = (float(cm[0,0]) / (cm[0,0] + cm[0,1])) 
print specificity
sensitivity = (float(cm[1,1]) / (cm[1,1] + cm[1,0])) 
print sensitivity

# grid searches for SVM - could still use some tuning
clf = svm.SVC()
C_range = range(1,5)
kernel_range = ['poly', 'rbf']
degree_range = range(1,6)
param_grid = dict(C=C_range, kernel=kernel_range, degree=degree_range)
grid = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')
grid.fit(X, y)
print grid.best_score_

clf = svm.SVC()
C_range = range(1,5)
kernel_range = ['poly', 'rbf']
degree_range = range(1,6)
param_grid = dict(C=C_range, kernel=kernel_range, degree=degree_range)
grid = GridSearchCV(clf, param_grid, cv=5, scoring='roc_auc')
grid.fit(X, y)
print grid.best_score_

clf = svm.SVC()
C_range = range(1, 20)
kernel_range = ['linear']
gamma_range = range(0,20)
param_grid = dict(C=C_range, kernel=kernel_range)
grid = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')
grid.fit(X, y)
print grid.best_score_

clf = svm.SVC()
C_range = range(1, 20)
kernel_range = ['linear']
gamma_range = range(0,20)
param_grid = dict(C=C_range, kernel=kernel_range)
grid = GridSearchCV(clf, param_grid, cv=5, scoring='roc_auc')
grid.fit(X, y)
print grid.best_score_

# generate preds for SVM using best parameters
best = grid.best_estimator_
best.fit(X_train, y_train)
y_pred_class = best.predict(X_test)

# SVM confusion matrix and specificity / sensitivity
print metrics.confusion_matrix(y_test, y_pred_class)
cm = metrics.confusion_matrix(y_test, y_pred_class)
specificity = (float(cm[0,0]) / (cm[0,0] + cm[0,1])) 
print specificity
sensitivity = (float(cm[1,1]) / (cm[1,1] + cm[1,0])) 
print sensitivity