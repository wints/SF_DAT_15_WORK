import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from collections import defaultdict
from sklearn import metrics
import statsmodels.formula.api as smf

# http://hypem.com/playlist/tags/dance/json/1/data.js
dance_dict = defaultdict(list)
for page in range(0,21):
    url = "http://hypem.com/playlist/tags/dance/json/"+str(page)+"/data.js"
    song_json = requests.get(url).json()
    del song_json['version']
    columns = ['mediaid', 'artist', 'title', 'dateposted', 'sitename', 'loved_count', 'posted_count', 'time']
    for column in columns:
        dance_dict[column] += [song[column] for song in song_json.values()]
        
dance1 = pd.DataFrame(dance_dict)
dance1.drop_duplicates(inplace=True)
dance1.dropna(inplace=True)
dance1['tag'] = 'dance'
dance1.to_csv('dance1.csv', encoding='utf-8', index=False)

# http://hypem.com/playlist/tags/electronic/json/1/data.js
elec_dict = defaultdict(list)
for page in range(0,21):
    url = "http://hypem.com/playlist/tags/electronic/json/"+str(page)+"/data.js"
    song_json = requests.get(url).json()
    del song_json['version']
    columns = ['mediaid', 'artist', 'title', 'dateposted', 'sitename', 'loved_count', 'posted_count', 'time']
    for column in columns:
        elec_dict[column] += [song[column] for song in song_json.values()]
        
elec1 = pd.DataFrame(elec_dict)
elec1.drop_duplicates(inplace=True)
elec1.dropna(inplace=True)
elec1['tag'] = 'electonric'
elec1.to_csv('elec1.csv', encoding='utf-8', index=False)

# http://hypem.com/playlist/tags/future%20garage/json/1/data.js
# this worked!
future_dict = defaultdict(list)
for page in range(0,21):
    url = "http://hypem.com/playlist/tags/future%20garage/json/"+str(page)+"/data.js"
    song_json = requests.get(url).json()
    del song_json['version']
    columns = ['mediaid', 'artist', 'title', 'dateposted', 'sitename', 'loved_count', 'posted_count', 'time']
    for column in columns:
        future_dict[column] += [song[column] for song in song_json.values()]
        
future1 = pd.DataFrame(future_dict)
future1.drop_duplicates(inplace=True)
future1.dropna(inplace=True)
future1['tag'] = 'future'
future1.to_csv('future1.csv', encoding='utf-8', index=False)

# http://hypem.com/playlist/tags/dubstep/json/1/data.js
dub_dict = defaultdict(list)
for page in range(0,21):
    url = "http://hypem.com/playlist/tags/dubstep/json/"+str(page)+"/data.js"
    song_json = requests.get(url).json()
    del song_json['version']
    columns = ['mediaid', 'artist', 'title', 'dateposted', 'sitename', 'loved_count', 'posted_count', 'time']
    for column in columns:
        dub_dict[column] += [song[column] for song in song_json.values()]
        
dub1 = pd.DataFrame(dub_dict)
dub1.drop_duplicates(inplace=True)
dub1.dropna(inplace=True)
dub1['tag'] = 'dubstep'
dub1.to_csv('dub1.csv', encoding='utf-8', index=False)

# http://hypem.com/playlist/tags/experimental/json/1/data.js
exp_dict = defaultdict(list)
for page in range(0,21):
    url = "http://hypem.com/playlist/tags/experimental/json/"+str(page)+"/data.js"
    song_json = requests.get(url).json()
    del song_json['version']
    columns = ['mediaid', 'artist', 'title', 'dateposted', 'sitename', 'loved_count', 'posted_count', 'time']
    for column in columns:
        exp_dict[column] += [song[column] for song in song_json.values()]
        
exp1 = pd.DataFrame(exp_dict)
exp1.drop_duplicates(inplace=True)
exp1.dropna(inplace=True)
exp1['tag'] = 'experimental'
exp1.to_csv('exp1.csv', encoding='utf-8', index=False)

# http://hypem.com/playlist/tags/funk/json/1/data.js
funk_dict = defaultdict(list)
for page in range(0,21):
    url = "http://hypem.com/playlist/tags/funk/json/"+str(page)+"/data.js"
    song_json = requests.get(url).json()
    del song_json['version']
    columns = ['mediaid', 'artist', 'title', 'dateposted', 'sitename', 'loved_count', 'posted_count', 'time']
    for column in columns:
        funk_dict[column] += [song[column] for song in song_json.values()]
        
funk1 = pd.DataFrame(funk_dict)
funk1.drop_duplicates(inplace=True)
funk1.dropna(inplace=True)
funk1['tag'] = 'funk'
funk1.to_csv('funk1.csv', encoding='utf-8', index=False)

# http://hypem.com/playlist/tags/hip%20hop/json/1/data.js
hiphop_dict = defaultdict(list)
for page in range(0,21):
    url = "http://hypem.com/playlist/tags/hip%20hop/json/"+str(page)+"/data.js"
    song_json = requests.get(url).json()
    del song_json['version']
    columns = ['mediaid', 'artist', 'title', 'dateposted', 'sitename', 'loved_count', 'posted_count', 'time']
    for column in columns:
        hiphop_dict[column] += [song[column] for song in song_json.values()]
        
hiphop1 = pd.DataFrame(hiphop_dict)
hiphop1.drop_duplicates(inplace=True)
hiphop1.dropna(inplace=True)
hiphop1['tag'] = 'hiphop'
hiphop1.to_csv('hiphop.csv', encoding='utf-8', index=False)

# http://hypem.com/playlist/tags/house/json/1/data.js
house_dict = defaultdict(list)
for page in range(0,21):
    url = "http://hypem.com/playlist/tags/house/json/"+str(page)+"/data.js"
    song_json = requests.get(url).json()
    del song_json['version']
    columns = ['mediaid', 'artist', 'title', 'dateposted', 'sitename', 'loved_count', 'posted_count', 'time']
    for column in columns:
        house_dict[column] += [song[column] for song in song_json.values()]
        
house1 = pd.DataFrame(house_dict)
house1.drop_duplicates(inplace=True)
house1.dropna(inplace=True)
house1['tag'] = 'house'
house1.to_csv('house.csv', encoding='utf-8', index=False)

# http://hypem.com/playlist/tags/instrumental/json/1/data.js
instrumental_dict = defaultdict(list)
for page in range(0,21):
    url = "http://hypem.com/playlist/tags/instrumental/json/"+str(page)+"/data.js"
    song_json = requests.get(url).json()
    del song_json['version']
    columns = ['mediaid', 'artist', 'title', 'dateposted', 'sitename', 'loved_count', 'posted_count', 'time']
    for column in columns:
        instrumental_dict[column] += [song[column] for song in song_json.values()]
        
instrumental1 = pd.DataFrame(instrumental_dict)
instrumental1.drop_duplicates(inplace=True)
instrumental1.dropna(inplace=True)
instrumental1['tag'] = 'instrumental'
instrumental1.to_csv('instrumental.csv', encoding='utf-8', index=False)

# http://hypem.com/playlist/tags/lo-fi/json/1/data.js
lofi_dict = defaultdict(list)
for page in range(0,21):
    url = "http://hypem.com/playlist/tags/lofi/json/"+str(page)+"/data.js"
    song_json = requests.get(url).json()
    del song_json['version']
    columns = ['mediaid', 'artist', 'title', 'dateposted', 'sitename', 'loved_count', 'posted_count', 'time']
    for column in columns:
        lofi_dict[column] += [song[column] for song in song_json.values()]
        
lofi1 = pd.DataFrame(lofi_dict)
lofi1.drop_duplicates(inplace=True)
lofi1.dropna(inplace=True)
lofi1['tag'] = 'lofi'
lofi1.to_csv('lofi.csv', encoding='utf-8', index=False)

# http://hypem.com/playlist/tags/pop/json/1/data.js
pop_dict = defaultdict(list)
for page in range(0,21):
    url = "http://hypem.com/playlist/tags/pop/json/"+str(page)+"/data.js"
    song_json = requests.get(url).json()
    del song_json['version']
    columns = ['mediaid', 'artist', 'title', 'dateposted', 'sitename', 'loved_count', 'posted_count', 'time']
    for column in columns:
        pop_dict[column] += [song[column] for song in song_json.values()]
        
pop1 = pd.DataFrame(pop_dict)
pop1.drop_duplicates(inplace=True)
pop1.dropna(inplace=True)
pop1['tag'] = 'pop'
pop1.to_csv('pop.csv', encoding='utf-8', index=False)

# http://hypem.com/playlist/tags/rock/json/1/data.js
rock_dict = defaultdict(list)
for page in range(0,21):
    url = "http://hypem.com/playlist/tags/rock/json/"+str(page)+"/data.js"
    song_json = requests.get(url).json()
    del song_json['version']
    columns = ['mediaid', 'artist', 'title', 'dateposted', 'sitename', 'loved_count', 'posted_count', 'time']
    for column in columns:
        rock_dict[column] += [song[column] for song in song_json.values()]
        
rock1 = pd.DataFrame(rock_dict)
rock1.drop_duplicates(inplace=True)
rock1.dropna(inplace=True)
rock1['tag'] = 'rock'
rock1.to_csv('rock.csv', encoding='utf-8', index=False)

# http://hypem.com/playlist/tags/singer-songwriter/json/1/data.js
singer_dict = defaultdict(list)
for page in range(0,21):
    url = "http://hypem.com/playlist/tags/singer-songwriter/json/"+str(page)+"/data.js"
    song_json = requests.get(url).json()
    del song_json['version']
    columns = ['mediaid', 'artist', 'title', 'dateposted', 'sitename', 'loved_count', 'posted_count', 'time']
    for column in columns:
        singer_dict[column] += [song[column] for song in song_json.values()]
        
singer1 = pd.DataFrame(singer_dict)
singer1.drop_duplicates(inplace=True)
singer1.dropna(inplace=True)
singer1['tag'] = 'singer-songwriter'
singer1.to_csv('singer.csv', encoding='utf-8', index=False)

all_cats = [dance1, elec1, future1, dub1, exp1, funk1, hiphop1, house1, instrumental1, lofi1, pop1, rock1, singer1]
songs =  pd.concat(all_cats, axis=0)
songs.to_csv('songs_categories_INSERT_DATE.csv', encoding='utf-8', index=False)



tag_dummies = songs['tag'].str.get_dummies()
tag_dummies_concat = (songs, tag_dummies)
songs = pd.concat(tag_dummies_concat, axis=1)

artist_dummies = songs['artist'].str.get_dummies()
artist_dummies_concat = (songs, artist_dummies)
songs = pd.concat(artist_dummies_concat, axis=1)

sitename_dummies = songs['sitename'].str.get_dummies()
sitename_dummies_concat = (songs, sitename_dummies)
songs = pd.concat(sitename_dummies_concat, axis=1)

sns.heatmap(songs.corr())

X, y = songs.drop(['mediaid', 'tag', 'time', 'title', 'sitename', 'loved_count', 'artist'], axis = 1), songs['loved_count']
songs.columns

X.shape
X_test.shape
y.shape
y.mean()
X.columns

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

import statsmodels.formula.api as smf

lm = smf.ols(formula='loved_count ~ posted_count + dateposted + dance + dubstep + electonric + experimental + funk + future + hiphop + house + instrumental + lofi + pop + rock', data=songs).fit()
lm.params
lm.conf_int()
lm.pvalues
lm.rsquared
y_pred = lm.predict(X_test)

import time 
int(time.time())


from sklearn.linear_model import LinearRegression
# instantiate and fit
linreg = LinearRegression()
linreg.fit(X_train, y_train)

y_pred = linreg.predict(X_test)
metrics.r2_score(y_pred, y_test)
print np.sqrt(metrics.mean_squared_error(y_test, y_pred))