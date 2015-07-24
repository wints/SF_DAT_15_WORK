import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


songs_all = pd.read_csv('/Users/wints/Desktop/ga_data_science/SF_DAT_15/project/songs_MASTER_DEDUPED_csv.csv') # creates songs_all dataframe
tags_raw = pd.read_csv('/Users/wints/Desktop/ga_data_science/SF_DAT_15/project/tags_list_MASTER.csv') # creates songs_all dataframe
tags_in = pd.read_csv('/Users/wints/Desktop/ga_data_science/SF_DAT_15/project/tags_list_MASTER_DUMMIES_CONCAT.csv') # creates songs_all dataframe

tags_in['tags'] = tags_in['tags'].str.replace("u'","'")
tags_in['tags'] = tags_in['tags'].str.replace("[","")
tags_in['tags'] = tags_in['tags'].str.replace("]","")

tags_in = tags_in[['mediaid', 'tags']]
songs_lim = songs_all[['mediaid', 'loved']]
tags_in.sort(columns='mediaid', inplace=True)
tags_in.reset_index(drop=True, inplace=True)
songs_lim.sort(columns='mediaid', inplace=True)
songs_lim.reset_index(drop=True, inplace=True)
comb = [tags_in, songs_lim]
tags_loved = pd.concat(comb, axis=1)
tags_loved.columns = ['mediaid', 'placeholder', 'tags', 'loved']
tags_loved = tags_loved[['mediaid', 'tags', 'loved']]


tags_true = tags_loved[tags_loved.loved == True]
tags_false = tags_loved[tags_loved.loved == False]
len(tags_true) # = 415
len(tags_false) # = 376
# total = 791

# tag cleaning code
tags_in = pd.read_csv('/Users/wints/Desktop/ga_data_science/SF_DAT_15/project/tags_list_MASTER_DUMMIES_CONCAT.csv') # creates songs_all dataframe
tags_in['tags'] = tags_in['tags'].str.replace("u'","'")
tags_in['tags'] = tags_in['tags'].str.replace("[","")
tags_in['tags'] = tags_in['tags'].str.replace("]","")
tags_in_dummies_2 = tags_in['tags'].str.get_dummies(sep=', ')
tags_in_dummies_2.to_csv('clean_tags_dummies_VICTORY_CLEANER.csv', index=False) 
def tagsum():
    tagcount = []
    for c in tags_in_dummies_2.columns:
        colsum = tags_in_dummies_2[c].sum()
        data = {}
        data['tag'] = c
        data['total'] = colsum
        tagcount.append(data)
    return tagcount
tags_count_dict = tagsum()
tags_count_df = pd.DataFrame(tags_count_dict)
tags_count_df.to_csv('tag_data_excel.csv', index=False)

# counts excel for loved = true
tags_true_dummies = tags_true['tags'].str.get_dummies(sep=', ')
tags_true = tags_true.merge(tags_true_dummies, right_index=True, left_index=True)
tags_true.drop(['tags', 'loved'], inplace=True, axis=1)

def tags_true_sum():
    tags_true_count = []
    for c in tags_true.columns:
        colsum = tags_true[c].sum()
        data = {}
        data['tag'] = c
        data['total'] = colsum
        tags_true_count.append(data)
    return tags_true_count
tags_true_dict = tags_true_sum()
tags_true_df = pd.DataFrame(tags_true_dict)
tags_true_df['tag_true_rate'] = (tags_true_df['total'] / 415)
tags_true_df.to_csv('tags_true_excel.csv', index=False)

# counts excel for loved = false
tags_false_dummies = tags_false['tags'].str.get_dummies(sep=', ')
tags_false = tags_false.merge(tags_false_dummies, right_index=True, left_index=True)
tags_false.drop(['tags', 'loved'], inplace=True, axis=1)

def tags_false_sum():
    tags_false_count = []
    for c in tags_false.columns:
        colsum = tags_false[c].sum()
        data = {}
        data['tag'] = c
        data['total'] = colsum
        tags_false_count.append(data)
    return tags_false_count
tags_false_dict = tags_false_sum()
tags_false_df = pd.DataFrame(tags_false_dict)
tags_false_df['tag_false_rate'] = (tags_false_df['total'] / 376)
tags_false_df.to_csv('tags_false_excel.csv', index=False)