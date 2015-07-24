import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns

songs_all = pd.read_csv('/Users/wints/Desktop/ga_data_science/SF_DAT_15/project/songs_MASTER_DEDUPED_csv.csv') # creates songs_all dataframe
songs = pd.read_csv('/Users/wints/Desktop/ga_data_science/SF_DAT_15/project/songs_MASTER_DEDUPED_csv.csv') # creates songs_all dataframe
tags_all = pd.read_csv('/Users/wints/Desktop/ga_data_science/SF_DAT_15/project/tags_list_MASTER_DUMMIES_CONCAT.csv') # creates songs_all dataframe
tags_all.reset_index(drop=True, inplace=True)
tags_all.head()
songs_all.head()
songs_all.describe()
songs_all.columns

songs_all.artist.value_counts()

sns.heatmap(songs_all.corr())

master_concat = (songs_all, tags_all)
songs = pd.concat(master_concat, axis=1)
# songs.to_csv('song_list_MASTER_TAGDUMMIES_2.csv', index=False)

# create an artist dummies frame 
artist_dummies = songs['artist'].str.get_dummies()
artist_dummies_concat = (songs, artist_dummies)
songs = pd.concat(artist_dummies_concat, axis=1)

sitename_dummies = songs['sitename'].str.get_dummies()
sitename_dummies_concat = (songs, sitename_dummies)
songs = pd.concat(sitename_dummies_concat, axis=1)

X, y = songs.drop(['loved', 'tags' , 'time', 'mediaid', 'title', 'sitename', 'loved_count', 'artist'], axis = 1), songs['loved_count']
X.shape
X_test.shape
y.shape
y.mean()
X.columns

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.linear_model import LinearRegression
# instantiate and fit
linreg = LinearRegression()
linreg.fit(X_train, y_train)

# calculate the R-squared value for the model
y_pred = linreg.predict(X_test)
metrics.r2_score(y_pred, y_test)
print np.sqrt(metrics.mean_squared_error(y_test, y_pred))

def train_test_rmse(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    y_pred = linreg.predict(X_test)
    return np.sqrt(metrics.mean_squared_error(y_test, y_pred))

train_test_rmse(X, y)
