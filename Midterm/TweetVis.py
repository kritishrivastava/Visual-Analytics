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
from bokeh.models import ColumnDataSource, HoverTool, Dropdown, PreText, Slider, Button, Label, Select, FactorRange
from bokeh.models import HoverTool, CustomJS,BoxZoomTool, ResetTool, BoxSelectTool, LassoSelectTool, TextInput
from bokeh.models.glyphs import Text
from bokeh.plotting import figure
from bokeh.layouts import widgetbox, gridplot, column, layout, row
# from bokeh.transform import factor_cmap
from bokeh.palettes import inferno

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

color_stream_1 = "Blue"
color_stream_2 = "Orange"

devices = ['Twitter for Android', 'Twitter for iPhone', 'Twitter Web Client']

# Global variable declaration
tf_stream1 = defaultdict(int)
tf_stream2 = defaultdict(int)
tf_normalized_stream1 = defaultdict(int)
tf_normalized_stream2 = defaultdict(int)
top_tf_normalized_stream1 = defaultdict(dict)
top_tf_normalized_stream2 = defaultdict(dict)
token_count_stream1 = 0
token_count_stream2 = 0
top_count = 50
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

doc = curdoc()

source_stream1 = defaultdict(int)
source_stream2 = defaultdict(int)

def insert_time_series_data_1(ts, msg, src):
    global df_tweet_1
    df_tweet_1.loc[len(df_tweet_1)] = [ts, msg, np.NaN]
    if src not in devices:
        return
    value = source_stream1.get(src)
    if value is None:
        source_stream1[src] = 1
    else:
        source_stream1[src] = value + 1

def insert_time_series_data_2(ts, msg, src):
    global df_tweet_2
    df_tweet_2.loc[len(df_tweet_2)] = [ts, msg, np.NaN]
    if src not in devices:
        return
    value = source_stream2.get(src)
    if value is None:
        source_stream2[src] = 1
    else:
        source_stream2[src] = value + 1

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
        insert_time_series_data_1(status.created_at, status.text, status.source)

    def on_error(self, status_code):
        if status_code == 420:
            # returning False in on_data disconnects the stream
            print("Hit rate limit")
            return False

class listener_2(tweepy.StreamListener):

    def on_status(self, status):
        nlp_2(status)
        insert_time_series_data_2(status.created_at, status.text, status.source)

    def on_error(self, status_code):
        if status_code == 420:
            # returning False in on_data disconnects the stream
            print("Hit rate limit")
            return False

def plot_tweet_rate(tweet_rate, tweet_rate_2):
    line1_datasource.data['x'] = tweet_rate['rounded_time'].tolist()
    line1_datasource.data['y'] = tweet_rate['count'].tolist()
    line1_datasource.trigger('data', line1_datasource.data, line1_datasource.data)

    line2_datasource.data['x'] = tweet_rate_2['rounded_time'].tolist()
    line2_datasource.data['y'] = tweet_rate_2['count'].tolist()
    line2_datasource.trigger('data', line2_datasource.data, line2_datasource.data)

def update_pie():
    # Updating Pie chart
    start_ang = float(token_count_stream1 / (token_count_stream2 + token_count_stream1))
    starts = [0, start_ang]
    ends = [start_ang, 1]
    starts = [i * 2 * 3.14 for i in starts]
    ends = [i * 2 * 3.14 for i in ends]
    pie_datasource.data['start_angle'] = starts
    pie_datasource.data['end_angle'] = ends
    pie_datasource.trigger('data', pie_datasource.data, pie_datasource.data)

def update_current_tweets():
    # Update the current tweets box
    text = "---------------- LATEST TWEETS ------------------\n\n\n "
    if len(all_tweets_stream1) > 20:
        for tweet in reversed(all_tweets_stream1[-20:]):
            text = text + tweet + "\n\n"
    else:
        for tweet in reversed(all_tweets_stream1):
            text = text + tweet + "\n\n"
    if len(all_tweets_stream2) > 20:
        for tweet in reversed(all_tweets_stream2[-20:]):
            text = text + tweet + "\n\n"
    else:
        for tweet in reversed(all_tweets_stream2):
            text = text + tweet + "\n\n"
    current_tweets_plot.text = text

def plot_word_cloud():
    word = []
    font_size = []
    x_rand = random.sample(range(1, 100), top_count)
    y_rand = random.sample(range(1, 100), top_count)
    dic = dict(top_tf_normalized_stream1)
    s = sum(list(dic.values()))
    for key, value in dic.items():
        word.append(key)
        val = (float)(value/s) * 100
        font_size.append("{0:.2f}".format(val) + 'pt')
    source_cloud_1.data['x'] = x_rand
    source_cloud_1.data['y'] = y_rand
    source_cloud_1.data['font_size'] = font_size
    source_cloud_1.data['word'] = word


def plot_word_cloud_2():
    x_rand_1 = random.sample(range(1, 100), top_count)
    y_rand_1 = random.sample(range(1, 100), top_count)
    word_1 = []
    font_size_1 = []
    dic = dict(top_tf_normalized_stream2)
    s = sum(list(dic.values()))
    for key, value in dic.items():
        word_1.append(key)
        val = (float)(value / s) * 100
        font_size_1.append("{0:.2f}".format(val) + 'pt')
    source_cloud_2.data['x'] = x_rand_1
    source_cloud_2.data['y'] = y_rand_1
    source_cloud_2.data['font_size'] = font_size_1
    source_cloud_2.data['word'] = word_1

def plot_source_bar_group():
    df1 = pd.DataFrame([source_stream1])
    df1['origin'] = 'Stream 1'
    df2 = pd.DataFrame([source_stream2])
    df2['origin'] = 'Stream 2'
    df1.append(df2)
    x1 = []
    count = []
    for key, value in source_stream1.items():
        x1.append((key, 'Term 1'))
        count.append(value)
    for key, value in source_stream2.items():
        x1.append((key, 'Term 2'))
        count.append(value)
    new = dict(x=x, counts=count)
    source_bar.stream(new, 20)

def update_scatter_plot():
    # Updating Scatter plots - sentiment and kmeans stream 1
    x, y, color_labels = clustering(all_tweets_stream1)
    source.data['x'] = x
    source.data['y'] = y
    colors = []
    kmeans_legend = []
    for label in color_labels:
        if label == 0:
            colors.append("Blue")
            kmeans_legend.append("Cluster 1")
        elif label == 1:
            colors.append("#FF9E00")
            kmeans_legend.append("Cluster 2")
        else:
            colors.append("Magenta")
            kmeans_legend.append("Cluster 3")
    source.data['kmeans_color_stream1'] = colors
    source.data['kmeans_legend'] = kmeans_legend
    colors_sentiment = []
    sentiment_legend = []
    for sentiment in sentiment_stream1:
        if sentiment == 0:
            colors_sentiment.append("#FFFF33")
            sentiment_legend.append("Neutral")
        elif sentiment > 0:
            colors_sentiment.append("#3cb44b")
            sentiment_legend.append("Positive")
        else:
            colors_sentiment.append("#e6194b")
            sentiment_legend.append("Negative")
    source.data['sentiment_color_stream1'] = colors_sentiment
    source.data['sentiment_legend'] = sentiment_legend
    source.data['tweet_text'] = all_tweets_stream1

    # Updating Scatter plots - sentiment and kmeans stream 2
    x2, y2, color_labels2 = clustering(all_tweets_stream2)
    source2.data['x'] = x2
    source2.data['y'] = y2
    colors2 = []
    kmeans_legend2 = []
    for label in color_labels2:
        if label == 0:
            colors2.append("Blue")
            kmeans_legend2.append("Cluster 1")
        elif label == 1:
            colors2.append("#FF9E00")
            kmeans_legend2.append("Cluster 2")
        else:
            colors2.append("Magenta")
            kmeans_legend2.append("Cluster 3")
    source2.data['kmeans_color_stream2'] = colors2
    source2.data['kmeans_legend'] = kmeans_legend2
    colors_sentiment2 = []
    sentiment_legend2 = []
    for sentiment in sentiment_stream2:
        if sentiment == 0:
            colors_sentiment2.append("#FFFF33")
            sentiment_legend2.append("Neutral")
        elif sentiment > 0:
            colors_sentiment2.append("#3cb44b")
            sentiment_legend2.append("Positive")
        else:
            colors_sentiment2.append("#e6194b")
            sentiment_legend2.append("Negative")
    source2.data['sentiment_color_stream2'] = colors_sentiment2
    source2.data['sentiment_legend'] = sentiment_legend2
    source2.data['tweet_text'] = all_tweets_stream2

    # Trigger change for each scatter plot
    sentiment_scatter1.data_source.trigger('data', sentiment_scatter1.data_source.data, sentiment_scatter1.data_source.data)
    sentiment_scatter2.data_source.trigger('data', sentiment_scatter2.data_source.data, sentiment_scatter2.data_source.data)
    kmeans_scatter1.data_source.trigger('data', kmeans_scatter1.data_source.data, kmeans_scatter1.data_source.data)
    kmeans_scatter2.data_source.trigger('data', kmeans_scatter2.data_source.data, kmeans_scatter2.data_source.data)

# Function to update all the visualizations after time step
def update_visualization():
    global df_tweet_1, df_tweet_2, pie_datasource
    ## Updating Line charts
    now = datetime.utcnow()
    min_time = now - timedelta(seconds=500)

    df_tweet_1.drop(df_tweet_1[df_tweet_1.Timestamp < min_time].index, inplace=True)
    df_tweet_1['rounded_time'] = df_tweet_1['Timestamp'].\
        apply(lambda x: x - timedelta(seconds=x.second - int(5 * round(float(x.second)/5))))

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
    plot_word_cloud_2()
    update_scatter_plot()
    update_current_tweets()
    # plot_source_bar_group()

def get_twitter_api_handle(consumer_key, consumer_secret, access_token, access_token_secret) :
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)
    print(api.me().name)
    return api


def create_twitter_stream(api, topic, callback):
    myStream = tweepy.Stream(auth=api.auth, listener=callback)
    myStream.filter(track=topic, async=True)


######## Creating Visualizations ###############

# Heading
heading = PreText(text="""\bANALYZING REAL TIME TWITTER DATA\b""", height=50, width=1500)

#------------------------------------------------------------------------------------------------------------------
# Catergory text search
search_1 = TextInput(value="default", title="Search Term 1:")
search_2 = TextInput(value="default", title="Search Term 2:")
button_go = Button(label="Compare", width=100, button_type="success")

#------------------------------------------------------------------------------------------------------------------
# Tweets display
text = "Real Time Tweets-- \n"
current_tweets_plot = PreText(text=text, width=400, height=900)

#------------------------------------------------------------------------------------------------------------------
# Line chart for tweet rate
df_tweet_1 = pd.DataFrame(columns=['Timestamp', 'Tweet', 'rounded_time'])
df_tweet_1 = df_tweet_1.fillna(0)
df_tweet_2 = pd.DataFrame(columns=['Timestamp', 'Tweet', 'rounded_time'])
df_tweet_2 = df_tweet_2.fillna(0)

tweet_rate_plot = figure(title='Tweet Rate(per 5 seconds)', x_axis_type="datetime", plot_width=900, plot_height=250,tools=[HoverTool()])
tweet_rate_plot.toolbar.logo = None
tweet_rate_plot.xaxis.axis_label = "Time(in Minute:Second format)"
tweet_rate_plot.yaxis.axis_label = "Count"
line1 = tweet_rate_plot.line(x=[], y=[], line_width=2, color=color_stream_1, legend='Search Term 1')
line1_datasource = line1.data_source
line2 = tweet_rate_plot.line(x=[], y=[], line_width=2, color = color_stream_2, legend='Search Term 2')
line2_datasource = line2.data_source

#------------------------------------------------------------------------------------------------------------------
# Pie chart for tweet division per category
colors = [color_stream_1, color_stream_2]
tweet_division_plot = figure(title='Total tweets Ratio', x_range=(-1, 1), y_range=(-1, 1),
                             plot_width=250, plot_height=250)
legend = ["Term 1","Term 2"]
pie = tweet_division_plot.wedge(x=0, y=0, radius=1, start_angle=[], end_angle=[], color=colors, tags = legend)
pie_datasource = pie.data_source
tweet_division_plot.axis.visible = False
tweet_division_plot.xgrid.grid_line_color = None
tweet_division_plot.ygrid.grid_line_color = None

#------------------------------------------------------------------------------------------------------------------
# Bar chart for tweets per device
device_tweet_plot = figure(plot_width=650, plot_height=300, title="Number of Tweets per device type")

# x = [('Twitter for Android', 'Stream 1'), ('Twitter for Android', 'Stream 2'),
#      ('Twitter for iPhone', 'Stream 1'), ('Twitter for iPhone', 'Stream 2'),
#      ('Twitter Web Client', 'Stream 1'), ('Twitter Web Client', 'Stream 2')]
# counts = [0, 0, 0, 0, 0, 0]
# streams = ['Stream 1', 'Stream 2']
# source_bar = ColumnDataSource(data=dict(x=x, counts=counts))
# device_tweet_plot = figure(x_range=FactorRange(*x), plot_width=600, plot_height=400,
#                             title="Number of Tweets per device type", toolbar_location=None, tools="")
# bar = device_tweet_plot.vbar(x='x', top='counts', width=0.9, source=source_bar, line_color="white",
#                              fill_color=factor_cmap('x', palette=[color_stream_1, color_stream_2], factors=streams, start=1, end=2))

#------------------------------------------------------------------------------------------------------------------
# Word clouds for most frequent words
x_rand = random.sample(range(1, 100), top_count)
y_rand = random.sample(range(1, 100), top_count)
word_cloud_stream_1 = figure(x_range=(-20, 120), y_range=(-20, 120),plot_width=450, plot_height=350, tools="", title="Most Frequent words for Term 1")
df_stream_1 = pd.DataFrame(pd.np.empty((top_count, 2)) * pd.np.nan, columns=['word', 'weight'])
df_stream_1['word'] = ''
df_stream_1['weight'] = 5
df_stream_1['font_size'] = df_stream_1['weight'].apply(lambda x: "{0:.2f}".format(x) + 'pt')
df_stream_1['x'] = x_rand
df_stream_1['y'] = y_rand
source_cloud_1 = ColumnDataSource(data = dict(x = x_rand, y= y_rand, word = df_stream_1['word'].tolist(), font_size = df_stream_1['font_size'].tolist()),)
word_cloud_stream_1.text(x='x', y='y', text='word', text_font_size = 'font_size', source=source_cloud_1, text_color=color_stream_1)
word_cloud_stream_1.axis.visible = False
word_cloud_stream_1.xgrid.grid_line_color = None
word_cloud_stream_1.ygrid.grid_line_color = None

word_cloud_stream_2 = figure(x_range=(-20, 120), y_range=(-20, 120), plot_width=450, plot_height=350, tools="", title="Most Frequent words for Term 2")
df_stream_2 = pd.DataFrame(pd.np.empty((top_count, 2)) * pd.np.nan, columns=['word', 'weight'])
df_stream_2['word'] = ''
df_stream_2['weight'] = 5
df_stream_2['font_size'] = df_stream_2['weight'].apply(lambda x: "{0:.2f}".format(x) + 'pt')
df_stream_2['x'] = x_rand
df_stream_2['y'] = y_rand
source_cloud_2 = ColumnDataSource(data = dict(x=x_rand, y=y_rand, word = df_stream_2['word'].tolist(), font_size = df_stream_2['font_size'].tolist()),)
word_cloud_stream_2.text(x='x', y='y', text='word', text_font_size = 'font_size', source=source_cloud_2, text_color=color_stream_2)
word_cloud_stream_2.axis.visible = False
word_cloud_stream_2.xgrid.grid_line_color = None
word_cloud_stream_2.ygrid.grid_line_color = None
#------------------------------------------------------------------------------------------------------------------
# Scatter plot for stream 1
scatterplot_width = 375
scatterplot_height = 325
source = ColumnDataSource(data=dict(
    x=[0,0,0],
    y=[0,0,0],
    sentiment_color_stream1= ["red","red",'white'],
    kmeans_color_stream1 = ["red","red",'white'],
    tweet_text=['a','a','a'],
    sentiment_legend = ["red","red",'white'],
    kmeans_legend = ["red","red",'white'],
))
hover = HoverTool(tooltips=[
    ("Tweet", "@tweet_text")
])
hover1 = HoverTool(tooltips=[
    ("Tweet", "@tweet_text")
])
# Scatter plot for clustering using sentiment - Category 1
sentiment_stream1_plot = figure(plot_width=scatterplot_width, plot_height=scatterplot_height, title="Clustering using sentiments on Term 1",
                                tools=[hover, LassoSelectTool(), ResetTool()])
sentiment_scatter1 = sentiment_stream1_plot.circle('x', 'y', size=5, fill_color="sentiment_color_stream1",
                                                   line_color=None, legend='sentiment_legend', source=source)
sentiment_stream1_plot.axis.visible = False
# sentiment_stream1_plot.xgrid.grid_line_color = None
# sentiment_stream1_plot.ygrid.grid_line_color = None

# Scatter plot for clustering using kmeans - Category 1
kmeans_stream1_plot = figure(plot_width=scatterplot_width, plot_height=scatterplot_height, title="MiniBatch-KMeans clustering on Term 1",
                             x_range=sentiment_stream1_plot.x_range, y_range=sentiment_stream1_plot.y_range, tools=[hover1, LassoSelectTool(), ResetTool()])
kmeans_scatter1 = kmeans_stream1_plot.circle('x', 'y', size=5, fill_color = 'kmeans_color_stream1',
                                             line_color=None, legend='kmeans_legend', source=source )
kmeans_stream1_plot.axis.visible = False
# kmeans_stream1_plot.xgrid.grid_line_color = None
# kmeans_stream1_plot.ygrid.grid_line_color = None


#------------------------------------------------------------------------------------------------------------------
# Scatter plot for stream 2
source2 = ColumnDataSource(data=dict(
    x=[0,0,0],
    y=[0,0,0],
    sentiment_color_stream2= ["red","red",'white'],
    kmeans_color_stream2 = ["red","red",'white'],
    tweet_text=['a','a','a'],
    sentiment_legend = ["red","red",'white'],
    kmeans_legend = ["red","red",'white'],
))
hover2 = HoverTool(tooltips=[
    ("Tweet", "@tweet_text")
])
hover21 = HoverTool(tooltips=[
    ("Tweet", "@tweet_text")
])

# Scatter plot for clustering using sentiment - Category 2
sentiment_stream2_plot = figure(plot_width=scatterplot_width, plot_height=scatterplot_height, title="Clustering using sentiments on Term 2",
                                tools=[hover2, LassoSelectTool(), ResetTool()])
sentiment_scatter2 = sentiment_stream2_plot.circle('x', 'y', size=5, fill_color="sentiment_color_stream2",
                                                   line_color=None, legend='sentiment_legend', source=source2)
sentiment_stream2_plot.axis.visible = False
# sentiment_stream2_plot.xgrid.grid_line_color = None
# sentiment_stream2_plot.ygrid.grid_line_color = None

# Scatter plot for clustering using kmeans - Category 2
kmeans_stream2_plot = figure(plot_width=scatterplot_width, plot_height=scatterplot_height, title="MiniBatch-KMeans clustering on Term 2",
                             x_range=sentiment_stream2_plot.x_range, y_range=sentiment_stream2_plot.y_range, tools=[hover21, LassoSelectTool(), ResetTool()])
kmeans_scatter2 = kmeans_stream2_plot.circle('x', 'y', size=5, fill_color = 'kmeans_color_stream2',
                                             line_color=None, legend='kmeans_legend', source=source2 )
kmeans_stream2_plot.axis.visible = False
# kmeans_stream2_plot.xgrid.grid_line_color = None
# kmeans_stream2_plot.ygrid.grid_line_color = None


#layout for the visualization
wgt_search = row(widgetbox(search_1), widgetbox(search_2), widgetbox(button_go))
l1 = layout([[tweet_rate_plot], [device_tweet_plot, tweet_division_plot],
                 [word_cloud_stream_1, word_cloud_stream_2],
                 [sentiment_stream1_plot,sentiment_stream2_plot],
                 [kmeans_stream1_plot,kmeans_stream2_plot]])
layout = layout([heading], [wgt_search], [l1, current_tweets_plot])

doc.add_root(layout)
doc.add_periodic_callback(update_visualization, 5000)

####### Twitter API calls ################

api_2 = get_twitter_api_handle(consumer_key_2, consumer_secret_2, access_token_2, access_token_secret_2)
topic = ['trump']
create_twitter_stream(api_2, topic, listener_2())

api_1 = get_twitter_api_handle(consumer_key_1, consumer_secret_1, access_token_1, access_token_secret_1)
topic = ['north korea']
create_twitter_stream(api_1, topic, listener_1())