from datetime import datetime, timedelta
import string
from threading import Timer
import time
from collections import defaultdict
from collections import Counter
import random
import tweepy
import warnings

import pandas as pd
import numpy as np
from pandas import Series, DataFrame, Panel

from nltk import word_tokenize
from nltk.corpus import stopwords
import preprocessor as p
from textblob import TextBlob

from bokeh.plotting import *
from bokeh.models import ColumnDataSource, HoverTool, Dropdown, PreText, Slider, Button, Label, Select
from bokeh.models.glyphs import Text
from bokeh.plotting import figure
from bokeh.layouts import widgetbox, gridplot, column, layout

from sklearn.feature_extraction.text import *
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans

# Ignore bokeh warnings
warnings.filterwarnings('ignore')
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

#Kriti's account
consumer_key_1="VcJ7LgeONhabS1o0b6CUfnZY2"
consumer_secret_1="wvkRYBPU9bem0EoJR04uvfnCWIFNlXusoGLca6b4vCCJWuWxsb"

access_token_1="921914312058384384-Uhqe3dePUkRuzjVKvAVPpoNlPclJMnh"
access_token_secret_1="crEaGnf626IpikzjxpwXopKvwzOqiYU6KRASF9phg4gaP"

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
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(data)
    reducedDimensions_X = TruncatedSVD(n_components=2).fit_transform(X_train_tfidf)
    x = [x[0] for x in reducedDimensions_X]
    y = [x[1] for x in reducedDimensions_X]
    mbk = MiniBatchKMeans(init='k-means++', n_clusters=3, batch_size=50, n_init=3, max_no_improvement=10, verbose=0).fit(reducedDimensions_X)
    return x, y,  mbk.labels_


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
    sentiment_stream1.append(sentiment)
    # if sentiment == 0:
    #     nuetral_stream1.append(status.text)
    # elif sentiment > 0:
    #     positive_stream1.append(status.text)
    # else:
    #     negative_stream1.append(status.text)


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
    sentiment_stream2.append(sentiment)
    # if sentiment == 0:
    #     nuetral_stream2.append(status.text)
    # elif sentiment > 0:
    #     positive_stream2.append(status.text)
    # else:
    #     negative_stream2.append(status.text)

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

def plot_tweet_rate(tweet_rate, tweet_rate_2):
    #tweet_rate_plot = figure(plot_width=800, plot_height=300)
    tweet_rate_plot.line(x=tweet_rate['rounded_time'].tolist(), y=tweet_rate['count'].tolist(), line_width=2)
    #layout.children[0] = tweet_rate_plot

    tweet_rate_plot.line(x=tweet_rate_2['rounded_time'].tolist(), y=tweet_rate_2['count'].tolist(), line_width=2)
    layout.children[0] = tweet_rate_plot

    # ds1.trigger('data', ds1.data, ds1.data)

def update_pie():
    # Updating Pie chart
    now = datetime.utcnow()
    # start_ang = float(now.second / 60)
    start_ang = float(token_count_stream1 / (token_count_stream2 + token_count_stream1))
    # end_ang = float(tweet_count_stream2 / (token_count_stream2 + token_count_stream1));
    starts = [0, start_ang]
    ends = [start_ang, 1]
    starts = [i * 2 * 3.14 for i in starts]
    ends = [i * 2 * 3.14 for i in ends]
    pie_datasource.data['start_angle'] = starts
    pie_datasource.data['end_angle'] = ends
    pie_datasource.trigger('data', pie_datasource.data, pie_datasource.data)


def plot_word_cloud():
    #print("Test word cloud")
    #print(top_tf_normalized_stream1)
    df = pd.DataFrame(top_tf_normalized_stream1, columns=['word', 'weight'])
    df['font_size'] = df['weight'].apply(lambda x: "{0:.2f}".format(x) + 'pt')
    #print(df.dtypes.index)
    #df = pd.DataFrame([top_tf_normalized_stream1], columns=top_tf_normalized_stream1.keys())
    x_rand = random.sample(range(1, 100), top_count)
    y_rand = random.sample(range(1, 100), top_count)
    df['x'] = x_rand
    df['y'] = y_rand
    source = ColumnDataSource(df)
    word_cloud_stream_1.text(x='x', y='y', text='word', text_font_size = 'font_size', source=source)
    #word_cloud_plot.add_glyph(source, glyph)
    layout.children[2] = word_cloud_stream_1

def update_scatter_plot():
    global clustered_source_stream1, sentiment_colors
    # Updating Scatter plots - sentiment and kmeans stream 1
    x, y, color_labels = clustering(all_tweets_stream1)
    clustered_source_stream1.data['x'] = x
    clustered_source_stream1.data['y'] = y
    kmeans_scatter1_datasource.data['x'] = x
    kmeans_scatter1_datasource.data['y'] = y
    colors = []
    for label in color_labels:
        if label == 0:
            colors.append("Blue")
        elif label == 1:
            colors.append("#FF9E00")
        else:
            colors.append("Magenta")
    kmeans_scatter1_datasource.data['fill_color'] = colors
    colors_sentiment = []
    for sentiment in sentiment_stream1:
        if sentiment == 0:
            colors_sentiment.append("#FFFF33")
        elif sentiment > 0:
            colors_sentiment.append("#3cb44b")
        else:
            colors_sentiment.append("#e6194b")
    clustered_source_stream1.data['sentiment_colors'] = colors_sentiment
    sentiment_colors = colors_sentiment
        # clustered_source_stream1.data['sentiment_colors']
    # Updating Scatter plots - sentiment and kmeans stream 2
    x2, y2, color_labels2 = clustering(all_tweets_stream2)
    sentiment_scatter2_datasource.data['x'] = x2
    sentiment_scatter2_datasource.data['y'] = y2
    kmeans_scatter2_datasource.data['x'] = x2
    kmeans_scatter2_datasource.data['y'] = y2
    colors2 = []
    for label in color_labels2:
        if label == 0:
            colors2.append("Blue")
        elif label == 1:
            colors2.append("#FF9E00")
        else:
            colors2.append("Magenta")
    kmeans_scatter2_datasource.data['fill_color'] = colors2
    colors_sentiment2 = []
    for sentiment in sentiment_stream2:
        if sentiment == 0:
            colors_sentiment2.append("#FFFF33")
        elif sentiment > 0:
            colors_sentiment2.append("#3cb44b")
        else:
            colors_sentiment2.append("#e6194b")
    sentiment_scatter2_datasource.data['fill_color'] = colors_sentiment2
    # Trigger change for each scatter plot
    clustered_source_stream1.trigger('data', clustered_source_stream1.data, clustered_source_stream1.data)
    sentiment_scatter2_datasource.trigger('data', sentiment_scatter2_datasource.data, sentiment_scatter2_datasource.data)
    kmeans_scatter1_datasource.trigger('data', kmeans_scatter1_datasource.data, kmeans_scatter1_datasource.data)
    kmeans_scatter2_datasource.trigger('data', kmeans_scatter2_datasource.data, kmeans_scatter2_datasource.data)

# Function to update all the visualizations after time step
def update_visualization():
    global df_tweet_1, df_tweet_2, pie_datasource
    ## Updating Line charts
    now = datetime.utcnow()
    min_time = now - timedelta(seconds=500)

    df_tweet_1.drop(df_tweet_1[df_tweet_1.Timestamp < min_time].index, inplace=True)
    df_tweet_1['rounded_time'] = df_tweet_1['Timestamp'].apply(lambda x: x - timedelta(seconds=x.second - round(x.second, -1)))

    tweet_rate = df_tweet_1.groupby(['rounded_time']).size()
    tweet_rate = tweet_rate.to_frame().reset_index()
    tweet_rate = tweet_rate.rename(columns={0: 'count'})

    df_tweet_2.drop(df_tweet_2[df_tweet_2.Timestamp < min_time].index, inplace=True)
    df_tweet_2['rounded_time'] = df_tweet_2['Timestamp'].apply(
        lambda x: x - timedelta(seconds=x.second - round(x.second, -1)))

    tweet_rate_2 = df_tweet_2.groupby(['rounded_time']).size()
    tweet_rate_2 = tweet_rate_2.to_frame().reset_index()
    tweet_rate_2 = tweet_rate_2.rename(columns={0: 'count'})

    # Calling update function for each graph
    plot_tweet_rate(tweet_rate, tweet_rate_2)
    update_pie()
    plot_word_cloud()
    update_scatter_plot()




def get_twitter_api_handle(consumer_key, consumer_secret, access_token, access_token_secret) :
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)
    print(api.me().name)
    return api


def create_twitter_stream(api, topic, callback):
    myStream = tweepy.Stream(auth=api.auth, listener=callback)
    myStream.filter(track=topic, async=True)


####### Twitter API calls ################

api_2 = get_twitter_api_handle(consumer_key_2, consumer_secret_2, access_token_2, access_token_secret_2)
topic = ['football']
create_twitter_stream(api_2, topic, listener_2())

api_1 = get_twitter_api_handle(consumer_key_1, consumer_secret_1, access_token_1, access_token_secret_1)
topic = ['north korea']
create_twitter_stream(api_1, topic, listener_1())


######## Creating Visualizations ###############

doc = curdoc()

# Heading
heading = PreText(text="""CLUSTERING ALGORITHM ON WHOLESALE CUSTOMERS DATA""", height=25, width=500)

# Catergory text search
search_1 = PreText(text="""\n\nCategory 1 """, height=50, width=200)
search_2 = PreText(text="""\n\nCategory 2 """, height=50, width=200)
button_go = Button(label="Evaluate", width=100, button_type="success")

# Tweets display
current_tweets_plot = PreText(text='Current tweets will go here-- \n Tweet numer 1\n Tweet numer 2\n', width=500, height=900)

# Line chart for tweet rate
columns = ['Timestamp', 'Tweet', 'rounded_time']
df_tweet_1 = pd.DataFrame(columns=columns)
df_tweet_1 = df_tweet_1.fillna(0)
columns = ['Timestamp', 'Tweet', 'rounded_time']
df_tweet_2 = pd.DataFrame(columns=columns)
df_tweet_2 = df_tweet_2.fillna(0)
tweet_rate_plot = figure(plot_width=800, plot_height=300)
line1 = tweet_rate_plot.line(x = [], y = [], line_width=2)
# tweet_rate_plot.title = "Comparing tweet rate"
ds1 = line1.data_source

# Pie chart for tweet division per category
colors = ["red", "green"]
tweet_division_plot = figure(x_range=(-1,1), y_range=(-1,1),plot_width=300, plot_height=300)
pie = tweet_division_plot.wedge(x=0, y=0, radius=1, start_angle=[], end_angle=[], color=colors)
pie_datasource = pie.data_source
tweet_division_plot.axis.visible = False


# Bar chart for tweets per device
device_tweet_plot = figure(plot_width=600, plot_height=400, title="Number of Tweets per device type")

# Word clouds for most frequent words
word_cloud_stream_1 = figure(x_range=(-20, 120), y_range=(-20, 120), plot_width=500, plot_height=500)
word_cloud_stream_2 = figure(x_range=(-20, 120), y_range=(-20, 120), plot_width=500, plot_height=500)

# Scatter plot for clustering using sentiment - Category 1
cluster_data_stream1 = {
    'x': [],
    'y': [],
    'kmeans_colors': [],
    'sentiment_colors': []
}
clustered_source_stream1 = ColumnDataSource(data=cluster_data_stream1)
scatterplot_width = 400
sentiment_colors = []
sentiment_stream1_plot = figure(plot_width=scatterplot_width, plot_height=scatterplot_width, title="Clustering using sentiments on Category 1")
sentiment_scatter1 = sentiment_stream1_plot.circle(x='x', y='y', size=5,  source = clustered_source_stream1, line_color=None)
# sentiment_scatter1_datasource = sentiment_scatter1.data_source
# fill_color = 'sentiment_colors'
sentiment_stream1_plot.axis.visible = False
sentiment_stream1_plot.xgrid.grid_line_color = None
sentiment_stream1_plot.ygrid.grid_line_color = None

# Scatter plot for clustering using sentiment - Category 2
sentiment_stream2_plot = figure(plot_width=scatterplot_width, plot_height=scatterplot_width, title="Clustering using sentiments on Category 2")
sentiment_scatter2 = sentiment_stream2_plot.circle(x=[], y=[], size=5, fill_color = [], line_color=None )
sentiment_scatter2_datasource = sentiment_scatter2.data_source
sentiment_stream2_plot.axis.visible = False
sentiment_stream2_plot.xgrid.grid_line_color = None
sentiment_stream2_plot.ygrid.grid_line_color = None

# Scatter plot for clustering using kmeans - Category 1
kmeans_stream1_plot = figure(plot_width=scatterplot_width, plot_height=scatterplot_width, title="MiniBatch-KMeans clustering on Category 1")
kmeans_scatter1 = kmeans_stream1_plot.circle(x=[], y=[], size=5, fill_color = [], line_color=None )
kmeans_scatter1_datasource = kmeans_scatter1.data_source
kmeans_stream1_plot.axis.visible = False
kmeans_stream1_plot.xgrid.grid_line_color = None
kmeans_stream1_plot.ygrid.grid_line_color = None

# Scatter plot for clustering using kmeans - Category 2
kmeans_stream2_plot = figure(plot_width=scatterplot_width, plot_height=scatterplot_width, title="MiniBatch-KMeans clustering on Category 2")
kmeans_scatter2 = kmeans_stream2_plot.circle(x=[], y=[], size=5, fill_color = [], line_color=None)
kmeans_scatter2_datasource = kmeans_scatter2.data_source
kmeans_stream2_plot.axis.visible = False
kmeans_stream2_plot.xgrid.grid_line_color = None
kmeans_stream2_plot.ygrid.grid_line_color = None

# Rendering all plots
layout = layout([heading], [search_1, search_2, button_go],
                [[[tweet_rate_plot], [device_tweet_plot, tweet_division_plot ],
                [word_cloud_stream_1, word_cloud_stream_2],
                [sentiment_stream1_plot,sentiment_stream2_plot],
                [kmeans_stream1_plot,kmeans_stream2_plot]],[current_tweets_plot]])
doc.add_root(layout)
doc.add_periodic_callback(update_visualization, 10000)



# x = np.linspace(-50, 150, 100)
# y = x
#
# print(top_tf_normalized_stream1)



# x_rand = random.sample(range(1, 100), 5)
# y_rand = random.sample(range(1, 100), 5)
#
# df = pd.DataFrame()
#
# df['word'] = ['a', 'b', 'c', 'd', 'e']
# df['x'] = x_rand
# df['y'] = y_rand
#
# print(x_rand, y_rand)
# source = ColumnDataSource(df)
#
# print(df)
#
# word_cloud_plot.text(x='x', y='y', text='word', source=source)
#
# layout.children[2] = word_cloud_plot

# duration is in seconds
#Timer(2, update_visualization).start()



#test = [('north', 6.481481481481481), ('korea', 6.172839506172839), ('rt', 4.320987654320987), ('open', 1.8518518518518516), ('letter', 1.8518518518518516), ('parliaments', 1.8518518518518516), ('number', 1.8518518518518516), ('countries', 1.8518518518518516), ('declares', 1.8518518518518516), ('power', 1.8518518518518516), ('…', 1.8518518518518516), ('``', 1.5432098765432098), ('full-fledged', 1.5432098765432098), ('nuclear', 1.5432098765432098), ("''", 1.5432098765432098), ('trump', 1.2345679012345678), ('cnni', 1.2345679012345678), ('’', 1.2345679012345678), ('window', 0.9259259259259258), ('diplomacy', 0.9259259259259258)]
