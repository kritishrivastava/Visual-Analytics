import tweepy
from bokeh.sampledata import us_states
from bokeh.plotting import *
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
from threading import Timer
import time

import pandas as pd
import numpy as np
from pandas import Series, DataFrame, Panel


#suhas acccount
consumer_key_1="UqbCeaE4OdRgZcgNGniiIL29h"
consumer_secret_1="JtQsLFnZGBX6MnEjBIwh8hQYPnGjHkVWYwCopQ4oKCfdBVolge"

access_token_1="921218524575600640-yYmqPHPT71JVoDkQK4LMPsY0AxtCuJ4"
access_token_secret_1="T560zV6EIOpu1ygf1bztfXPj7WGUAuNgfjooqgRLgGi10"

#anirudh account
consumer_key_2="7JV3cDSA2mVLuDd1oPG2k9F1F"
consumer_secret_2="7lrYE7uiMnMyjs0gbGOIXwz8c0fX7Otg8I5aOKuombb7H0TV4c"

access_token_2="2273168329-px91XwVztXVPjrDGvvyKuQLCyOFi8Zd19NzakMP"
access_token_secret_2="tN1RKG0v8jtaIfvoXOGeUry7v0IcPRPLSK5x3qf6UQu4C"


columns = ['Timestamp', 'Tweet']

pd.set(max_rows=5) # this limit maximum numbers of rows

df_tweet_1 = pd.DataFrame(columns=columns)
df_tweet_1 = df_tweet_1.fillna(0)

words_stream_1 = []
words_stream_2 = []

def process_tweet(tweet):
    stop = stopwords.words('english') + list(string.punctuation)
    words = [i for i in word_tokenize(tweet.lower()) if i not in stop]
    return words

class listener_1(tweepy.StreamListener):

    def on_status(self, status):
        global df_tweet_1
        #print(process_tweet(status.text))
        df_tweet_1.append({'Timestamp': status.created_at, 'Tweet': status.text}, ignore_index=True)
        #print(status.text)
        # if status.coordinates is not None:
        #     xc = status.coordinates['coordinates'][0]
        #     yc = status.coordinates['coordinates'][1]
        #     print(xc, yc)

    def on_error(self, status_code):
        if status_code == 420:
            # returning False in on_data disconnects the stream
            print("Hit rate limit")
            return False

class listener_2(tweepy.StreamListener):

    def on_status(self, status):
        print("listerner_2")

    def on_error(self, status_code):
        if status_code == 420:
            # returning False in on_data disconnects the stream
            print("Hit rate limit")
            return False

def update_visualization():
    print(df_tweet_1)
    #plot()
    Timer(2, update_visualization).start()


def get_twitter_api_handle(consumer_key, consumer_secret, access_token, access_token_secret) :
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    api = tweepy.API(auth)

    print(api.me().name)

    return api


def create_twitter_stream(api, topic, callback):
    myStream = tweepy.Stream(auth=api.auth, listener=callback)
    myStream.filter(track=topic, async=True)


def plot():
    x = 1

api_2 = get_twitter_api_handle(consumer_key_2, consumer_secret_2, access_token_2, access_token_secret_2)

topic = ['football']
create_twitter_stream(api_2, topic, listener_2())

api_1 = get_twitter_api_handle(consumer_key_1, consumer_secret_1, access_token_1, access_token_secret_1)
topic = ['baseball']
create_twitter_stream(api_1, topic, listener_1())


# duration is in seconds
Timer(2, update_visualization).start()

