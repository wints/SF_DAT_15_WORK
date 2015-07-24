import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns

songs_all = pd.read_csv('/Users/wints/Desktop/ga_data_science/SF_DAT_15/project/songs_MASTER_INDEXED.csv') # creates songs_all dataframe

songs_all.head()
songs_all.describe()
songs_all.columns

sns.pairplot(songs_all, x_vars=['posted_count','loved_count', 'artist_index_mod', 'site_index_mod'], y_vars='loved', size=4.5, aspect=0.7)

# good code
sns.lmplot(x='loved_count', y='loved', data=songs_all, ci=None) # some positive relationship
plt.scatter(songs_all.loved_count, songs_all.loved)
songs_all.plot(kind='scatter', x='loved_count', y='loved')

sns.lmplot(x='posted_count', y='loved', data=songs_all, ci=None) # some negative relationship
plt.scatter(songs_all.posted_count, songs_all.loved)
songs_all.plot(kind='scatter', x='posted_count', y='loved')

sns.lmplot(x='artist_index_mod', y='loved', data=songs_all, ci=None) # some negative relationship
plt.scatter(songs_all.posted_count, songs_all.loved)
songs_all.plot(kind='scatter', x='artist_index_mod', y='loved')

sns.lmplot(x='site_index_mod', y='loved', data=songs_all, ci=None) # some negative relationship
plt.scatter(songs_all.posted_count, songs_all.loved)
songs_all.plot(kind='scatter', x='site_index_mod', y='loved')

sns.lmplot(x='time', y='loved', data=songs_all, ci=None) # some negative relationship
plt.scatter(songs_all.time, songs_all.loved)
songs_all.plot(kind='scatter', x='time', y='loved')
# end good code

sns.pairplot(songs_all)