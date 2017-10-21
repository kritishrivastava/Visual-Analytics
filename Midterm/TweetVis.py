import tweepy
from bokeh.sampledata import us_states
from bokeh.plotting import *
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
from threading import Timer
import time
from collections import defaultdict
from collections import Counter
import preprocessor as p
from textblob import TextBlob
from sklearn.cluster import MiniBatchKMeans
import pandas as pd
import numpy as np
from pandas import Series, DataFrame, Panel

# Setting options for tweet preprocessing - remove URL and emoji
p.set_options(p.OPT.URL, p.OPT.EMOJI)

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

words_stream_1 = []
words_stream_2 = []

# Global variable declaration
tf_stream1 = defaultdict(int)
tf_stream2 = defaultdict(int)
tf_normalized_stream1 = defaultdict(int)
tf_normalized_stream2 = defaultdict(int)
top_tf_normalized_stream1 = defaultdict(dict)
top_tf_normalized_stream2 = defaultdict(dict)
token_count_stream1 = 0
token_count_stream2 = 0
top_count = 20
sentiment_stream1 = []
sentiment_stream2 = []
positive_stream1 = []
negative_stream1 = []
nuetral_stream1 = []
positive_stream2 = []
negative_stream2 = []
nuetral_stream2 = []
all_tweets_stream1 = []
all_tweets_stream2 = []
kmeans_data_stream1 = []
kmeans_data_stream2 = []

def process_tweet(tweet):
    tweet = p.clean(tweet)
    stop = stopwords.words('english') + list(string.punctuation)
    words = [i for i in word_tokenize(tweet.lower()) if i not in stop]
    return words

def clustering(data):
    mbk = MiniBatchKMeans(init='k-means++', n_clusters=3, batch_size=50, n_init=3, max_no_improvement=10, verbose=0).fit(data)
    return mbk.labels_



class listener_1(tweepy.StreamListener):

    def on_status(self, status):
        global df_tweet_1
        global token_count_stream1
        global tf_stream1
        global tf_normalized_stream1
        global top_tf_normalized_stream1
        global top_count
        global sentiment_stream1
        global positive_stream1
        global negative_stream1
        global nuetral_stream1
        global all_tweets_stream1
        global kmeans_data_stream1

        all_tweets_stream1.append(status.text)
        # Process tweets and update term frequencies
        list_of_tokens = process_tweet(status.text)
        token_count_stream1 = token_count_stream1 + len(list_of_tokens)
        for token in list_of_tokens:
            tf_stream1[token] += 1
        for token, count in tf_stream1.items():
            tf_normalized_stream1[token] = float(count / token_count_stream1) * 100
        # Update the top k most frequent words
        top_tf_normalized_stream1 = Counter(tf_normalized_stream1).most_common(top_count)
        # Finding sentiment of the tweet
        sentiment = (status.text).sentiment.polarity
        sentiment_stream1.append(status.text, sentiment)
        if sentiment == 0:
            nuetral_stream1.append(status.text)
        elif sentiment > 0:
            positive_stream1.append(status.text)
        else:
            negative_stream1.append(status.text)
        # Updating data from kmeans clustering
        kmeans_data_stream1.append((status.text, status.created_at, status.source, status.lang))

        df_tweet_1.append({'Timestamp': status.created_at, 'Tweet': status.text}, ignore_index=True)
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
        # Using global variables
        global token_count_stream2
        global tf_stream2
        global tf_normalized_stream2
        global top_tf_normalized_stream2
        global top_count
        global sentiment_stream2
        global positive_stream2
        global negative_stream2
        global nuetral_stream2
        global all_tweets_stream2
        global kmeans_data_stream2

        all_tweets_stream2.append(status.text)
        #Process tweets and update term frequencies
        list_of_tokens = process_tweet(status.text)
        token_count_stream2 = token_count_stream2 + len(list_of_tokens)
        for token in list_of_tokens:
            tf_stream2[token] += 1
        for token, count in tf_stream2.items():
            tf_normalized_stream2[token] = float(count/token_count_stream2)*100
        # Update the top k most frequent words
        top_tf_normalized_stream2 = Counter(tf_normalized_stream2).most_common(top_count)
        # Finding sentiment of the tweet
        sentiment = TextBlob(status.text).sentiment.polarity
        sentiment_stream2.append((status.text, sentiment))
        if sentiment == 0:
            nuetral_stream2.append(status.text)
        elif sentiment > 0:
            positive_stream2.append(status.text)
        else:
            negative_stream2.append(status.text)
        # Updating data from kmeans clustering
        kmeans_data_stream2.append((status.text, status.created_at, status.source, status.lang))

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

# api_1 = get_twitter_api_handle(consumer_key_1, consumer_secret_1, access_token_1, access_token_secret_1)
# topic = ['baseball']
# create_twitter_stream(api_1, topic, listener_1())


# duration is in seconds
# Timer(2, update_visualization).start()