import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns

songs_all = pd.read_csv('/Users/wints/Desktop/ga_data_science/SF_DAT_15/project/songs_MASTER_INDEXED.csv') # creates songs_all dataframe

songs_all.head()
songs_all.describe()
songs_all.columns

# good code til end (no tag concat for this stage)
X, y = songs_all.drop(['loved', 'time' ,  'mediaid', 'title', 'sitename', 'artist'], axis = 1), songs_all['loved']
X.shape
y.shape
X.columns

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# putting together a logistic regression model
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred_class = logreg.predict(X_test)
from sklearn import metrics
print metrics.accuracy_score(y_test, y_pred_class)

y_all_no = [0] * len(y_test)
y_all_yes = [1] * len(y_test)
np.mean(y_all_yes == y_test) # 54.04%
np.mean(y_all_no == y_test) # 45.95%

prds = logreg.predict(X)
print metrics.confusion_matrix(y_test, y_pred_class)

sensitivity = (float(cm[0,0]) / (cm[0,0] + cm[0,1])) 
specificity = (float(cm[1,1]) / (cm[1,1] + cm[1,0])) 

from sklearn.cross_validation import cross_val_score
scores = cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=10)
print scores
print scores.mean()