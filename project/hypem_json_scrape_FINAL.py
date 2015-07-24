import requests
import pandas as pd
from collections import defaultdict

new_dict = defaultdict(list)
for page in range(0,15):
    url = "http://hypem.com/playlist/history/wints/json/"+str(page)+"/data.js"
    song_json = requests.get(url).json()
    del song_json['version']
    columns = ['mediaid', 'artist', 'title', 'dateposted' , 'sitename', 'loved_count', 'posted_count', 'time']
    for column in columns:
        new_dict[column] += [song[column] for song in song_json.values()]
        
        
songs_test = pd.DataFrame(new_dict)
songs_test.to_csv('hypem_hist_only_pull.csv', encoding='utf-8', index=False)


ids = []
for page in range(1,12):
    loved_dict = defaultdict(list)
    url = "http://hypem.com/playlist/loved/wints/json/"+str(page)+"/data.js"
    song_json = requests.get(url).json()
    del song_json['version']
    ids += [song['mediaid'] for song in song_json.values()]

len(ids)

def loved(media_id):
    return media_id in ids

songs_test['loved'] = songs_test['mediaid'].apply(loved)
songs_test_2['loved'] = songs_test['mediaid'].apply(loved)

songs_test.loved.value_counts()
songs_test_2.loved.value_counts()

songs_test.to_csv('songs_test_plus_loved.csv', encoding='utf-8')
songs_test.to_csv('songs_test_GOOD_DATASET.csv', encoding='utf-8', index=False)

loved_dict = defaultdict(list)
for page in range(0,100):
    url = "http://hypem.com/playlist/loved/wints/json/"+str(page)+"/data.js"
    loved_json = requests.get(url).json()
    del loved_json['version']
    columns = ['mediaid', 'artist', 'title', 'sitename', 'loved_count', 'posted_count', 'time']
    for column in columns:
        loved_dict[column] += [song[column] for song in loved_json.values()]

lf = pd.DataFrame(loved_dict)
lf.shape
lf.to_csv('loved_all_v2.csv', encoding='utf-8')
lf['loved'] = True

songs_all = pd.read_csv('songs_1070.csv')
loved_all = pd.read_csv('loved_all.csv')

songs_comb = [songs_all, loved_all]

songs_master = pd.concat(songs_comb)
songs_master.describe

songs_master.to_csv('songs_MASTER_v2.csv', encoding='utf-8')

songs_master = songs_master.reset_index(drop=True)

songs_master_deduped = songs_master.drop_duplicates(subset='mediaid')
songs_master_deduped.loved.value_counts()

songs_master_deduped.reset_index(drop=True)
songs_master_deduped.to_csv('songs_MASTER_DEDUPED_v2.csv', encoding='utf-8')

songids = songs_master_deduped['mediaid'].tolist()

