import requests
import pandas as pd
from bs4 import BeautifulSoup
from HTMLParser import HTMLParser
from collections import defaultdict

h = HTMLParser()

h.unescape(requests.get('http://hypem.com/track/rhs4').text) # sample response - raw

# songid list setup
songs_all = pd.read_csv('songs_MASTER_DEDUPED.csv') # creates songs_all dataframe
songids = songs_all['mediaid'].tolist() # creates list of songids for use in function
print songids

# function to pull tags for a single song
def songtags(id_):
        r = h.unescape(requests.get('https://hypem.com/track/'+id_).text)
        soup = BeautifulSoup(r)
        return  {"id": id_, "tags": [tag.text for tag in soup.find('ul',{'class':\
        "tags"}).findChildren('a')]}
songtags('2c1g8')

# my first attempt at a for loop
# getting AttributeError: 'NoneType' object has no attribute 'findChildren'
# means that I am hitting a song with no tags and not handling properly
tags_dict = {}
def songtags_list_1(songids):
    for id_ in songids:    
        r = h.unescape(requests.get('https://hypem.com/track/'+id_).text)
        soup = BeautifulSoup(r)
        tags_dict['id'] = id_
        if soup.find('ul',{'class': "tags"}) != None:
            tags_dict['tags'] = [tag.text for tag in soup.find('ul',{'class':"tags"})\
            .findChildren('a')]
    return tags_dict
songtags_list_1(songids)

# initial feedback from stack exchange question - doesn't quite work
# UnboundLocalError: local variable 'tag' referenced before assignment    
# UPDATe: after edits, the function works but is returning the last song only
# to resolve at office hours today

# include a sleep function while crawling, e.g. 3 seconds; can also randomize the sleep

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
    return tags_data
    
songids_short = songids[0:100]
tags_data = songtags_list_2(songids_short)
pd.DataFrame(tags_data)

# my own attempt to modify function with a rule in place for missing 'ul'
# i get the same NameError: global name 'tags' is not defined
tags_dict = {}
def songtags_list_3(songids):
    for id_ in songids:    
        r = h.unescape(requests.get('https://hypem.com/track/'+id_).text)
        soup = BeautifulSoup(r)
        tags_dict['id'] = id_
        tags_dict['tags'] = [child.text for child in tags.findChildren('a') for tag in\
        [soup.find('ul',{'class': "tags"})] if 'ul' != None]
        return tags_dict

songtags_list_3(songids)

# ok, ditching the list comprehension and writing another if statement instead
# this runs forever but ONLY returns values for the last id_ in the list
# this is my best bet so far...
tags_dict = {}
def songtags_list_4(songids):
    for id_ in songids:    
        r = h.unescape(requests.get('https://hypem.com/track/'+id_).text)
        soup = BeautifulSoup(r)
        tags_dict['id'] = id_
        if soup.find('ul',{'class': "tags"}) != None:        
            for tag in soup.find('ul',{'class': "tags"}).findChildren('a'):        
                tags_dict['tags'] = tag.text
        return tags_dict
                
songtags_list_4(songids)
     

# adding an if statement to (hopefully) skip if tags not present
# doesn't return a complete dictionary, though : only the first id_ in the list
        
def songtags_list_5(songids):
    for id_ in songids: 
        r = h.unescape(requests.get('https://hypem.com/track/'+id_).text)
        soup = BeautifulSoup(r)
        tags_dict = {"id": id_, "tags": [tag.text for tag in soup.find('ul',{'class':\
        "tags"}).findChildren('a') if soup.find('ul',{'class':\
        "tags"}) != None]}
        return tags_dict
        
songtags_list_5(songids)

# adding an if statement to (hopefully) skip if tags not present
# doesn't return a complete dictionary, though : only the first id_ in the list

def songtags_list_6(songids):
    for id_ in songids: 
        r = h.unescape(requests.get('https://hypem.com/track/'+id_).text)
        soup = BeautifulSoup(r)
        if soup.find('ul',{'class': "tags"}) != None:
            return  {"id": id_, "tags": [tag.text for tag in soup.find('ul',{'class':\
            "tags"}).findChildren('a')]}        
songtags_list_6(songids)