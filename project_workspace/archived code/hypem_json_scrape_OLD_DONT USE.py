# history: 13 pages, http://hypem.com/playlist/history/wints/json/1/data.js
# loved : 11 pages, http://hypem.com/playlist/loved/wints/json/1/data.js

import requests
import urllib, json
import pandas as pd
import urlopen
from bs4 import BeautifulSoup

histpages = range(0,14)
histpages
def gethistory(pages):
    for i in pages:
        r = requests.get('http://hypem.com/playlist/history/wints/json/i/data.js')
    return data

histdata = gethistory(histpages)
histdata

r = requests.get('http://hypem.com/playlist/history/wints/json/1/data.js') # sample response - raw

import json
import urllib2

url = "http://hypem.com/playlist/history/wints/json/1/data.js"
data = json.load(urllib2.urlopen(url))

def gethistory(pages):
    for i in pages:
        url = 'http://hypem.com/playlist/history/wints/json/i/data.js'
        data = json.load(urllib2.urlopen(url))
        # data = urllib2.urlopen(url)
    return data

histdata = gethistory(histpages)
histdata

def gethistory(pages):
    for i in pages:
        r = requests.get('http://hypem.com/playlist/history/wints/json/i/data.js')
        data = r.json()
        # data = urllib2.urlopen(url)
    return data

h.unescape(requests.get('http://hypem.com/track/rhs4').text) # sample response - raw

histfile = open('hypem_history_manscrape.json', 'rU')
histframe = pd.read_json(histfile)
