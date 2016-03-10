#!/usr/bin/env python
# -*- coding: utf-8 -*-

#for saving objects to files and reading them back in
import pickle

#for retrieving and slicing the twitter stream
import simplejson
from requests_oauthlib import OAuth1
from itertools import islice, chain
import requests

#for natural language processing
from nltk.corpus import stopwords

#for statistics on tweeted words
from collections import Counter
import heapq

#for regular expressions
import re

#for scheduling at a given time
import datetime



"""Follow the twitter stream in a given time window and extract 
    tweets with keywords for word statistics
    """


#I/O of python objects via pickle files
def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

#access the twitter stream with this function
def tweet_generator():
    stream = requests.post('https://stream.twitter.com/1.1/statuses/filter.json',
                           auth=auth,
                           stream=True,
                           data={"locations" : US_BOUNDING_BOX, #set API filters (see twitter website)
                                 "language" : "en",
                                 "filter_level": "none"})
        
    for line in stream.iter_lines():
        if not line:  # filter out keep-alive new lines
            continue
        tweet = simplejson.loads(line)
        if 'text' in tweet:
            yield tweet['text']


#get the n largest elements in an array
def nlargest(n, word_scores):
    return heapq.nlargest(n, word_scores, key=lambda x: x[1])




#initialize the twitter stream
with open("twitter_secrets.json.nogit") as fh:
    secrets = simplejson.loads(fh.read())

auth = OAuth1(
              secrets["api_key"],
              secrets["api_secret"],
              secrets["access_token"],
              secrets["access_token_secret"]
              )

#Pacific west of Mexico to Atlantic east of Canada
US_BOUNDING_BOX = "-125.00,24.94,-66.93,49.59"


#use filter words
stop = set(stopwords.words('english'))

#initialize the dictionaries
counter_1 = Counter()
counter_2 = Counter()
counter_3 = Counter()
counter_4 = Counter()


#define keywords
search_word_1 = "trump"
search_word_2 = "rubio"
search_word_3 = "cruz"
search_word_4 = "kasich"


#create dummy word list
safe_words = ['abwasserversorgung', 'anhalter', 'gruppendynamik', 'altersheim', 'lilaloe']

for word in safe_words:
    counter_1[word] += 1
    counter_2[word] += 1
    counter_3[word] += 1
    counter_4[word] += 1

#initialize counter for tries
k = 0

#loop for reading the stream
while True:
    #set stop time
    if datetime.datetime.now() > datetime.datetime(2016, 3, 4, 6, 00, 0):
        break
    #set start time
    elif datetime.datetime.now() < datetime.datetime(2016, 3, 4, 3, 00, 0):
        pass
    #try taking data
    else:
        print(k, datetime.datetime.now())
        try:
            for tweet in islice(tweet_generator(), 1000):
                words = re.sub(r'[.,_@#!?&:;"/+-]', r' ', tweet.lower())
                if search_word_1 in words.split():  #check here for keyword
                    for word in words.split():
                        if word not in stop:
                            counter_1[word] += 1
                if search_word_2 in words.split():  #check here for keyword
                    for word in words.split():
                        if word not in stop:
                            counter_2[word] += 1
                if search_word_3 in words.split():  #check here for keyword
                    for word in words.split():
                        if word not in stop:
                            counter_3[word] += 1
                if search_word_4 in words.split():  #check here for keyword
                    for word in words.split():
                        if word not in stop:
                            counter_4[word] += 1
            #save 1000 most often used words in files
            save_obj(nlargest(1000, counter_1.items()), search_word_1)
            save_obj(nlargest(1000, counter_2.items()), search_word_2)
            save_obj(nlargest(1000, counter_3.items()), search_word_3)
            save_obj(nlargest(1000, counter_4.items()), search_word_4)
            print("\n" + str(nlargest(10, counter_1.items())) + "\n")
            k = k + 1
        except:
            pass




