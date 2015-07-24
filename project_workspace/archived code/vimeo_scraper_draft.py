import requests
from bs4 import BeautifulSoup
from HTMLParser import HTMLParser

h = HTMLParser()

def getInfoForVideo(id_):
    r = h.unescape(requests.get('https://vimeo.com/'+id_).text)
    soup = BeautifulSoup(r)
    print "title:  ", soup.find('h1', {'class':'js-clip_title'}).text
    print "author:   ", re.search('Produced by: ([A-Z]\w+ [A-Z]\w+)', r).group(1)

h.unescape(requests.get('https://vimeo.com/53044579').text) # sample response - raw
    
    
getInfoForVideo('53044579') # calling the function for an ID