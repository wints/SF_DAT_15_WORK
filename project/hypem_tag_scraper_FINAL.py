import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from HTMLParser import HTMLParser
from collections import defaultdict
import time

h = HTMLParser()

h.unescape(requests.get('http://hypem.com/track/1m1ef').text) # sample response - raw

# songid list setup
songs_all = pd.read_csv('/Users/wints/Desktop/ga_data_science/SF_DAT_15/project/songs_MASTER_DEDUPED_csv.csv') # creates songs_all dataframe
songids = songs_all['mediaid'].tolist() # creates list of songids for use in function
print songids

# function to pull tags for a single mediaid
def songtags(id_):
        r = h.unescape(requests.get('https://hypem.com/track/'+id_).text)
        soup = BeautifulSoup(r)
        return  {"id": id_, "tags": [tag.text for tag in soup.find('ul',{'class':\
        "tags"}).findChildren('a')]}
songtags('23c5d')

# function for a list of mediaids and dataframe creation
# good code
def songtags_list_2(songids):
    tags_data = []
    for id_ in songids:    
        r = h.unescape(requests.get('https://hypem.com/track/'+id_).text)
        soup = BeautifulSoup(r)
        data = {}
        data['mediaid'] = id_
        data['tags'] = [child.text
                     for tag in [soup.find('ul',{'class': "tags"})]
                     if tag
                     for child in tag.findChildren('a')]
        tags_data.append(data)
        time.sleep(10)
    return tags_data

tags_df = songtags_list_2(tags_data)    
# end good code

#short list of songids for testing script
songids_short = songids[0:10]

# breaking the larger songids list into 
songids_1 = songids[0:25]
songids_2 = songids[25:50]
songids_3 = songids[50:75]
songids_4 = songids[75:100]
songids_5 = songids[100:125]
songids_6 = songids[125:150]
songids_7 = songids[150:175]
songids_8 = songids[175:200]
songids_9 = songids[200:225]
songids_10 = songids[225:250]
songids_11 = songids[250:275]
songids_12 = songids[275:300]
songids_13 = songids[300:325]
songids_14 = songids[325:350]
songids_15 = songids[350:375]
songids_16 = songids[375:400]
songids_17 = songids[400:425]
songids_18 = songids[425:450]
songids_19 = songids[450:475]
songids_20 = songids[475:500]
songids_21 = songids[500:525]
songids_22 = songids[525:550]
songids_23 = songids[550:575]
songids_24 = songids[575:600]
songids_25 = songids[600:625]
songids_26 = songids[625:650]
songids_27 = songids[650:675]
songids_28 = songids[675:700]
songids_29 = songids[700:725]
songids_30 = songids[725:750]
songids_31 = songids[750:775]
songids_32 = songids[775:793]

# songids 1-4
tags_data_1 = songtags_list_2(songids_1)
tags_frame_1 = pd.DataFrame(tags_data_1)

tags_data_2 = songtags_list_2(songids_2)
tags_frame_2 = pd.DataFrame(tags_data_2)

tags_data_3 = songtags_list_2(songids_3)
tags_frame_3 = pd.DataFrame(tags_data_3)

tags_data_4 = songtags_list_2(songids_4)
tags_frame_4 = pd.DataFrame(tags_data_4)

frames_1_4 = [tags_frame_1, tags_frame_2, tags_frame_3, tags_frame_4]
tags_frame_1_4 = pd.concat(frames_1_4)
tags_frame_1_4.to_csv('tags_list_1_4.csv', index=False)

# songids 5-8
tags_data_5 = songtags_list_2(songids_5)
tags_frame_5 = pd.DataFrame(tags_data_5)

tags_data_6 = songtags_list_2(songids_6)
tags_frame_6 = pd.DataFrame(tags_data_6)

tags_data_7 = songtags_list_2(songids_7)
tags_frame_7 = pd.DataFrame(tags_data_7)

tags_data_8 = songtags_list_2(songids_8)
tags_frame_8 = pd.DataFrame(tags_data_8)

frames_5_8 = [tags_frame_5, tags_frame_6, tags_frame_7, tags_frame_8]
tags_frame_5_8 = pd.concat(frames_5_8)
tags_frame_5_8.to_csv('tags_list_5_8.csv', index=False)

# songids 9-12
tags_data_9 = songtags_list_2(songids_9)
tags_frame_9 = pd.DataFrame(tags_data_9)
time.sleep(120)

tags_data_10 = songtags_list_2(songids_10)
tags_frame_10 = pd.DataFrame(tags_data_10)
time.sleep(120)

tags_data_11 = songtags_list_2(songids_11)
tags_frame_11 = pd.DataFrame(tags_data_11)
time.sleep(120)

tags_data_12 = songtags_list_2(songids_12)
tags_frame_12 = pd.DataFrame(tags_data_12)
time.sleep(120)

frames_9_12 = [tags_frame_9, tags_frame_10, tags_frame_11, tags_frame_12]
tags_frame_9_12 = pd.concat(frames_9_12)
tags_frame_9_12.to_csv('tags_list_9_12.csv', index=False)

# songids 13-16
tags_data_13 = songtags_list_2(songids_13)
tags_frame_13 = pd.DataFrame(tags_data_13)
time.sleep(120)

tags_data_14 = songtags_list_2(songids_14)
tags_frame_14 = pd.DataFrame(tags_data_14)
time.sleep(120)

tags_data_15 = songtags_list_2(songids_15)
tags_frame_15 = pd.DataFrame(tags_data_15)
time.sleep(120)

tags_data_16 = songtags_list_2(songids_16)
tags_frame_16 = pd.DataFrame(tags_data_16)
time.sleep(120)

frames_13_16 = [tags_frame_13, tags_frame_14, tags_frame_15, tags_frame_16]
tags_frame_13_16 = pd.concat(frames_13_16)
tags_frame_13_16.to_csv('tags_list_13_16.csv', index=False)

# songids 17-20
tags_data_17 = songtags_list_2(songids_17)
tags_frame_17 = pd.DataFrame(tags_data_17)
time.sleep(120)

tags_data_18 = songtags_list_2(songids_18)
tags_frame_18 = pd.DataFrame(tags_data_18)
time.sleep(120)

tags_data_19 = songtags_list_2(songids_19)
tags_frame_19 = pd.DataFrame(tags_data_19)
time.sleep(120)

tags_data_20 = songtags_list_2(songids_20)
tags_frame_20 = pd.DataFrame(tags_data_20)
time.sleep(120)

frames_17_20 = [tags_frame_17, tags_frame_18, tags_frame_19, tags_frame_20]
tags_frame_17_20 = pd.concat(frames_17_20)
tags_frame_17_20.to_csv('tags_list_17_20.csv', index=False)

# songids 21-24
tags_data_21 = songtags_list_2(songids_21)
tags_frame_21 = pd.DataFrame(tags_data_21)
time.sleep(120)

tags_data_22 = songtags_list_2(songids_22)
tags_frame_22 = pd.DataFrame(tags_data_22)
time.sleep(120)

tags_data_23 = songtags_list_2(songids_23)
tags_frame_23 = pd.DataFrame(tags_data_23)
time.sleep(120)

tags_data_24 = songtags_list_2(songids_24)
tags_frame_24 = pd.DataFrame(tags_data_24)
time.sleep(120)

frames_21_24 = [tags_frame_21, tags_frame_22, tags_frame_23, tags_frame_24]
tags_frame_21_24 = pd.concat(frames_21_24)
tags_frame_21_24.to_csv('tags_list_21_24.csv', index=False)

# songids 25-28
tags_data_25 = songtags_list_2(songids_25)
tags_frame_25 = pd.DataFrame(tags_data_25)
time.sleep(120)

tags_data_26 = songtags_list_2(songids_26)
tags_frame_26 = pd.DataFrame(tags_data_26)
time.sleep(120)

tags_data_27 = songtags_list_2(songids_27)
tags_frame_27 = pd.DataFrame(tags_data_27)
time.sleep(120)

tags_data_28 = songtags_list_2(songids_28)
tags_frame_28 = pd.DataFrame(tags_data_28)
time.sleep(120)

frames_25_28 = [tags_frame_25, tags_frame_26, tags_frame_27, tags_frame_28]
tags_frame_25_28 = pd.concat(frames_25_28)
tags_frame_25_28.to_csv('tags_list_25_28.csv', index=False)

# songids 29-32
tags_data_29 = songtags_list_2(songids_29)
tags_frame_29 = pd.DataFrame(tags_data_29)
time.sleep(120)

tags_data_30 = songtags_list_2(songids_30)
tags_frame_30 = pd.DataFrame(tags_data_30)
time.sleep(120)

tags_data_31 = songtags_list_2(songids_31)
tags_frame_31 = pd.DataFrame(tags_data_31)
time.sleep(120)

tags_data_32 = songtags_list_2(songids_32)
tags_frame_32 = pd.DataFrame(tags_data_32)
time.sleep(120)

frames_29_32 = [tags_frame_29, tags_frame_30, tags_frame_31, tags_frame_32]
tags_frame_29_32 = pd.concat(frames_29_32)
tags_frame_29_32.to_csv('tags_list_29_32.csv', index=False)


subframes = [tags_frame_1, tags_frame_2, tags_frame_3, tags_frame_4, tags_frame_5,\
 tags_frame_6, tags_frame_7, tags_frame_8, tags_frame_9, tags_frame_10, tags_frame_11, tags_frame_12,\
 tags_frame_13, tags_frame_14, tags_frame_15, tags_frame_16, tags_frame_17, tags_frame_18,\
 tags_frame_19, tags_frame_20, tags_frame_21, tags_frame_22, tags_frame_23, tags_frame_24,\
tags_frame_25, tags_frame_26, tags_frame_27, tags_frame_28, tags_frame_29, tags_frame_30,\
 tags_frame_31, tags_frame_32]
 
tags_frame_ALL_1 = pd.concat(subframes)
tags_frame_ALL_1.reset_index(drop=True, inplace=True)
tags_frame_ALL_1.to_csv('tags_ALL_v1.csv', index=False)
tags_frame_ALL_1.columns


tags_frame_ALL_1['tag_missing'] = tags_frame_ALL_1.tags.str.len() == 0
songids_new = tags_frame_ALL_1.mediaid[tags_frame_ALL_1['tag_missing'] == True]
len(songids_new)
songids_new.tolist()
frame_1_clean = tags_frame_ALL_1[tags_frame_ALL_1['tag_missing'] == False]

songids_new_1 = songids_new[0:25]
songids_new_2 = songids_new[25:50]
songids_new_3 = songids_new[50:75]
songids_new_4 = songids_new[75:100]
songids_new_5 = songids_new[100:125]
songids_new_6 = songids_new[125:150]
songids_new_7 = songids_new[150:159]

# songids 1-4
tags_data_1n = songtags_list_2(songids_new_1)
tags_frame_1n = pd.DataFrame(tags_data_1n)
time.sleep(120)

tags_data_2n = songtags_list_2(songids_new_2)
tags_frame_2n = pd.DataFrame(tags_data_2n)
time.sleep(120)

tags_data_3n = songtags_list_2(songids_new_3)
tags_frame_3n = pd.DataFrame(tags_data_3n)
time.sleep(120)

tags_data_4n = songtags_list_2(songids_new_4)
tags_frame_4n = pd.DataFrame(tags_data_4n)


frames_1_4n = [tags_frame_1n, tags_frame_2n, tags_frame_3n, tags_frame_4n]
tags_frame_1_4n = pd.concat(frames_1_4n)
tags_frame_1_4n.to_csv('tags_list_1_4n.csv', index=False)


# songids 5-7
tags_data_5n = songtags_list_2(songids_new_5)
tags_frame_5n = pd.DataFrame(tags_data_5n)
time.sleep(120)

tags_data_6n = songtags_list_2(songids_new_6)
tags_frame_6n = pd.DataFrame(tags_data_6n)
time.sleep(120)

tags_data_7n = songtags_list_2(songids_new_7)
tags_frame_7n = pd.DataFrame(tags_data_7n)

frames_5_7n = [tags_frame_5n, tags_frame_6n, tags_frame_7n]
tags_frame_5_7n = pd.concat(frames_5_7n)
tags_frame_5_7n.to_csv('tags_list_5_7n.csv', index=False)

subframes_2 = [tags_frame_1n, tags_frame_2n, tags_frame_3n, tags_frame_4n, tags_frame_5n,\
 tags_frame_6n, tags_frame_7n]
 
tags_frame_ALL_2 = pd.concat(subframes_2)
tags_frame_ALL_2.reset_index(drop=True, inplace=True)
tags_frame_ALL_2.to_csv('tags_ALL_v2.csv', index=False)
tags_frame_ALL_2.columns

tags_frame_ALL_2['tag_missing'] = tags_frame_ALL_2.tags.str.len() == 0
tags_frame_ALL_2.tag_missing.value_counts()
frame_2_clean = tags_frame_ALL_2[tags_frame_ALL_2['tag_missing'] == False]
len(songids_new_2)
songids_new_2.tolist()

# good code
def songtags_list_long(songids):
    tags_data = []
    for id_ in songids:    
        r = h.unescape(requests.get('https://hypem.com/track/'+id_).text)
        soup = BeautifulSoup(r)
        data = {}
        data['mediaid'] = id_
        data['tags'] = [child.text
                     for tag in [soup.find('ul',{'class': "tags"})]
                     if tag
                     for child in tag.findChildren('a')]
        tags_data.append(data)
        time.sleep(60)
    return tags_data

tags_data_last79 = songtags_list_long(songids_new_2)
frame_3_all = pd.DataFrame(tags_data_last79)
tags_frame_last79.to_csv('tags_ALL_v3.csv', index=False)

frame_1_clean, frame_2_clean, frame_3_all
all_3_frames = [frame_1_clean, frame_2_clean, frame_3_all]
master_tags_frame = pd.concat(all_3_frames)
master_tags_frame.columns
master_tags_frame_clean = master_tags_frame.drop('tag_missing', axis=1)

master_tags_frame_clean.reset_index(drop=True, inplace=True)
master_tags_frame_clean.to_csv('tags_list_MASTER.csv', index=False)

master_tags_dummies = master_tags_frame_clean['tags'].str.get_dummies(sep=',', dummy_na=True)
master_tags_dummies

concat_frames = (master_tags_frame_clean, master_tags_dummies)
master_tags_concatenated = pd.concat(concat_frames, axis=1)

master_tags_concatenated.to_csv('tags_list_MASTER_DUMMIES_CONCAT.csv', index=False)