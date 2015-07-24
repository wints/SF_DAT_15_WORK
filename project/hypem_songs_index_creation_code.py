import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns

songs_all = pd.read_csv('/Users/wints/Desktop/ga_data_science/SF_DAT_15/project/songs_MASTER_DEDUPED_csv.csv') # creates songs_all dataframe

songs_all['artist_index_mod'] = (np.where(songs_all[['artist', 'loved']].groupby('artist')\
.transform('sum') > 1 , songs_all[['artist', 'loved']].groupby('artist')\
.transform('sum') - 1 , 0)) / (np.where(songs_all[['artist', 'loved']].groupby('artist').\
transform('count') > 1, songs_all[['artist', 'loved']].groupby('artist').\
transform('count') -1, 0))

songs_all['site_index_mod'] = (np.where(songs_all[['sitename', 'loved']].groupby('sitename')\
.transform('sum') > 1 , songs_all[['sitename', 'loved']].groupby('sitename')\
.transform('sum') - 1 , 0)) / (np.where(songs_all[['artist', 'loved']].groupby('artist').\
transform('count') > 1, songs_all[['sitename', 'loved']].groupby('sitename').\
transform('count') -1, 0))

songs_all['artist_index_mod'].replace([np.inf, -np.inf], np.nan, inplace=True)
songs_all['site_index_mod'].replace([np.inf, -np.inf], np.nan, inplace=True)
songs_all['artist_index_mod'].fillna(value=0, inplace=True)
songs_all['site_index_mod'].fillna(value=0, inplace=True)
songs_all['artist'].fillna(value='artist_unknown', inplace=True)

songs_all.to_csv('songs_INDEXED_PLACEHOLDER.csv', index=False)

'''
ORIGINAL APPROACH - DON'T USE

artist_loved_rate = (songs_all[['artist', 'loved']].groupby('artist').transform('sum')) / (songs_all[['artist', 'loved']].groupby('artist').transform('count'))
site_loved_rate = (songs_all[['sitename', 'loved']].groupby('sitename').sum()) / (songs_all[['sitename', 'loved']].groupby('sitename').count())

songs_all['artist_index'] = (songs_all[['artist', 'loved']].groupby('artist').transform('sum')) / (songs_all[['artist', 'loved']].groupby('artist').transform('count'))
songs_all['site_index'] = (songs_all[['sitename', 'loved']].groupby('sitename').transform('sum')) / (songs_all[['sitename', 'loved']].groupby('sitename').transform('count'))
songs_all.artist_index.fillna(value=0, inplace=True)
'''

# create a master data file where nothing has been  made into a dummy
master_concat = (songs_all, tags_all)
songs = pd.concat(master_concat, axis=1)
songs.to_csv('song_list_MASTER_TAGDUMMIES_2.csv', index=False)

# create an artist dummies frame 
artist_dummies = songs['artist'].str.get_dummies()
artist_dummies_concat = (songs, artist_dummies)
songs = pd.concat(artist_dummies_concat, axis=1)

sitename_dummies = songs['sitename'].str.get_dummies()
sitename_dummies_concat = (songs, sitename_dummies)
songs = pd.concat(sitename_dummies_concat, axis=1)

songs.to_csv('songs_master_tag_artist_sitename_dummies.csv', index=False)