week_dict = defaultdict(list)
for page in range(0,10):
    url = "http://hypem.com/playlist/popular/week/json/"+str(page)+"/data.js"
    song_json = requests.get(url).json()
    del song_json['version']
    columns = ['mediaid', 'artist', 'title', 'dateposted' , 'sitename', 'loved_count', 'posted_count', 'time']
    for column in columns:
        week_dict[column] += [song[column] for song in song_json.values()]