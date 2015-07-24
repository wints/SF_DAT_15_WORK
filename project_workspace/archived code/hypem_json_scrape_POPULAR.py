import requests
import pandas as pd
from collections import defaultdict

# this week
# http://hypem.com/playlist/popular/week/json/1/data.js
week_dict = defaultdict(list)
for page in range(0,10):
    url = "http://hypem.com/playlist/popular/week/json/"+str(page)+"/data.js"
    song_json = requests.get(url).json()
    del song_json['version']
    columns = ['mediaid', 'artist', 'title', 'dateposted' , 'sitename', 'loved_count', 'posted_count', 'time']
    for column in columns:
        week_dict[column] += [song[column] for song in song_json.values()]
 
songs_week1 = pd.DataFrame(week_dict)
# songs_week.to_csv('week_7.14.15.csv', encoding='utf-8', index=False)
songs_week1.to_csv('week_7.15.15.csv', encoding='utf-8', index=False)
songs_week2.to_csv('week_7.16.15.csv', encoding='utf-8', index=False)
       
# last week
# http://hypem.com/playlist/popular/lastweek/json/1/data.js
lastweek_dict = defaultdict(list)
for page in range(0,10):
    url = "http://hypem.com/playlist/popular/lastweek/json/"+str(page)+"/data.js"
    song_json = requests.get(url).json()
    del song_json['version']
    columns = ['mediaid', 'artist', 'title', 'dateposted' , 'sitename', 'loved_count', 'posted_count', 'time']
    for column in columns:
        lastweek_dict[column] += [song[column] for song in song_json.values()]
 
songs_lastweek1 = pd.DataFrame(lastweek_dict)
# songs_lastweek.to_csv('lastweek_7.14.15.csv', encoding='utf-8', index=False)
# songs_lastweek1.to_csv('lastweek_7.15.15.csv', encoding='utf-8', index=False)
songs_lastweek2.to_csv('lastweek_7.16.15.csv', encoding='utf-8', index=False)

# remixes
# http://hypem.com/playlist/popular/remix/json/1/data.js
remixes_dict = defaultdict(list)
for page in range(0,10):
    url = "http://hypem.com/playlist/popular/remix/json/"+str(page)+"/data.js"
    song_json = requests.get(url).json()
    del song_json['version']
    columns = ['mediaid', 'artist', 'title', 'dateposted' , 'sitename', 'loved_count', 'posted_count', 'time']
    for column in columns:
        remixes_dict[column] += [song[column] for song in song_json.values()]

 
songs_remixes1 = pd.DataFrame(remixes_dict)
# songs_remixes.to_csv('remixes_7.15.15.csv', encoding='utf-8', index=False)
songs_remixes1.to_csv('remixes_7.16.15.csv', encoding='utf-8', index=False)

print requests.get('http://hypem.com/playlist/popular/remix/json/1/data.js').json()
# no remixes
# http://hypem.com/playlist/popular/noremix/json/1/data.js
noremixes_dict = defaultdict(list)
for page in range(0,10):
    url = "http://hypem.com/playlist/popular/noremix/json/"+str(page)+"/data.js"
    song_json = requests.get(url).json()
    del song_json['version']
    columns = ['mediaid', 'artist', 'title', 'dateposted' , 'sitename', 'loved_count', 'posted_count', 'time']
    for column in columns:
        noremixes_dict[column] += [song[column] for song in song_json.values()]
 
songs_noremixes = pd.DataFrame(noremixes_dict)
songs_noremixes.to_csv('noremixes_7.15.15.csv', encoding='utf-8', index=False)
songs_noremixes1.to_csv('noremixes_7.16.15.csv', encoding='utf-8', index=False)



songs_set = [songs_week, songs_week1, songs_lastweek, songs_lastweek1]
songs_4 = pd.concat(songs_set)
songs_4.mediaid.duplicated().sum()

http://hypem.com/playlist/search/popular/json/1/data.js