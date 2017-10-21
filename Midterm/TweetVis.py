import tweepy
from bokeh.plotting import *
from datetime import datetime, timedelta
from bokeh.plotting import figure
from bokeh.layouts import widgetbox, gridplot, column

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
total_tweet_count = 0
tweet_count_stream1= 0
tweet_count_stream2 = 0

def insert_time_series_data_1(ts, msg):
    global df_tweet_1
    df_tweet_1.loc[len(df_tweet_1)] = [ts, msg, np.NaN]

def process_tweet(tweet):
    tweet = p.clean(tweet)
    stop = stopwords.words('english') + list(string.punctuation)
    words = [i for i in word_tokenize(tweet.lower()) if i not in stop]
    return words

def clustering(data):
    mbk = MiniBatchKMeans(init='k-means++', n_clusters=3, batch_size=50, n_init=3, max_no_improvement=10, verbose=0).fit(data)
    return mbk.labels_


def nlp_1(status):
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
    global tweet_count_stream1

    # Update tweet count
    tweet_count_stream1 += 1
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
    sentiment = TextBlob(status.text).sentiment.polarity
    sentiment_stream1.append((status.text, sentiment))
    if sentiment == 0:
        nuetral_stream1.append(status.text)
    elif sentiment > 0:
        positive_stream1.append(status.text)
    else:
        negative_stream1.append(status.text)
    # Updating data from kmeans clustering
    kmeans_data_stream1.append((status.text, status.created_at, status.source, status.lang))


def nlp_2(status):
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
    global tweet_count_stream2

    # Update tweet counts
    tweet_count_stream2 += 1
    all_tweets_stream2.append(status.text)
    # Process tweets and update term frequencies
    list_of_tokens = process_tweet(status.text)
    token_count_stream2 = token_count_stream2 + len(list_of_tokens)
    for token in list_of_tokens:
        tf_stream2[token] += 1
    for token, count in tf_stream2.items():
        tf_normalized_stream2[token] = float(count / token_count_stream2) * 100
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

class listener_1(tweepy.StreamListener):

    def on_status(self, status):
        nlp_1(status)
        insert_time_series_data_1(status.created_at, status.text)

    def on_error(self, status_code):
        if status_code == 420:
            # returning False in on_data disconnects the stream
            print("Hit rate limit")
            return False

class listener_2(tweepy.StreamListener):

    def on_status(self, status):
        nlp_2(status)

    def on_error(self, status_code):
        if status_code == 420:
            # returning False in on_data disconnects the stream
            print("Hit rate limit")
            return False

# Function to update all the visualizations after time step
def update_visualization():
    global df_tweet_1
    ## Updating Line charts
    now = datetime.utcnow()
    min_time = now - timedelta(seconds=500)
    df_tweet_1.drop(df_tweet_1[df_tweet_1.Timestamp < min_time].index, inplace=True)
    df_tweet_1['rounded_time'] = df_tweet_1['Timestamp'].apply(lambda x: x - timedelta(seconds=x.second - round(x.second, -1)))

    tweet_rate = df_tweet_1.groupby(['rounded_time']).size()
    tweet_rate = tweet_rate.to_frame().reset_index()
    tweet_rate = tweet_rate.rename(columns={0: 'count'})

    ds1.data['x'] = tweet_rate['rounded_time'].tolist()
    ds1.data['y'] = tweet_rate['count'].tolist()

    # Updating Pie chart
    percents = [float(token_count_stream1 / (token_count_stream2 + token_count_stream1)) * 100,
                float(tweet_count_stream2 / (token_count_stream2 + token_count_stream1)) * 100]
    # percents = [0,0.15,0.4,0.7,1.0]
    starts = [0, 0.2]
    ends = [0.2, 1]
    print(starts)
    print(ends)
    print(percents)
    pie_datasource['start_angle'] = starts
    pie_datasource['end_angle'] = ends


    # Trigger all plot updates
    ds1.trigger('data', ds1.data, ds1.data)
    # pie_datasource.trigger('data', pie_datasource.data, pie_datasource.data)
    # pie_datasource.trigger('change')


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

doc = curdoc()

# Line chart for tweet rate
columns = ['Timestamp', 'Tweet', 'rounded_time']
df_tweet_1 = pd.DataFrame(columns=columns)
df_tweet_1 = df_tweet_1.fillna(0)
tweet_rate_plot = figure(plot_width=800, plot_height=300)
line1 = tweet_rate_plot.line(x = [], y = [], line_width=2)
ds1 = line1.data_source


# Pie chart for tweet category percentage
# percents = [float(token_count_stream1/total_tweet_count)*100,float(tweet_count_stream2/total_tweet_count)*100]
# starts = [p*2*np.pi for p in percents[:-1]]
# ends = [p*2*np.pi for p in percents[1:]]
percents = [0,0.7,1.0]
starts = [0, 0.7]
ends = [0.7, 1]
colors = ["red", "green", "blue", "orange", "yellow"]
tweet_division_plot = figure(x_range=(-1,1), y_range=(-1,1),plot_width=200, plot_height=200)
pie = tweet_division_plot.wedge(x=0, y=0, radius=1, start_angle=starts, end_angle=ends, color=colors)
pie_datasource = pie.data_source


# Rendering all plots
layout = column(tweet_rate_plot, tweet_division_plot)
doc.add_root(layout)


# api_2 = get_twitter_api_handle(consumer_key_2, consumer_secret_2, access_token_2, access_token_secret_2)
#
# topic = ['football']
# create_twitter_stream(api_2, topic, listener_2())

api_1 = get_twitter_api_handle(consumer_key_1, consumer_secret_1, access_token_1, access_token_secret_1)
topic = ['north korea']
create_twitter_stream(api_1, topic, listener_1())


# duration is in seconds
# Timer(2, update_visualization).start()

doc.add_periodic_callback(update_visualization, 10000)