import string
import time
from datetime import datetime, date, timedelta
import random

import re
#import simplejson
import operator
from collections import defaultdict
import warnings

from bokeh.sampledata import us_states
from bokeh.io import curdoc
from bokeh.models import ColumnDataSource, HoverTool, Dropdown, PreText, Slider, Button, Label, Select, DateRangeSlider
from bokeh.layouts import widgetbox, gridplot, column, layout, row
from bokeh.models import HoverTool, CustomJS,BoxZoomTool, ResetTool, BoxSelectTool, LassoSelectTool, TextInput
from bokeh.plotting import figure

from nltk import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem.snowball import SnowballStemmer

from itertools import chain

import preprocessor as p

import pandas as pd

import geocoder
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

from gensim.models import Word2Vec

us_states = us_states.data.copy()

default_search_1 = "sick"

flag_play = False

del us_states["HI"]
del us_states["AK"]

# Ignore warnings
warnings.filterwarnings('ignore')

p.set_options(p.OPT.EMOJI,p.OPT.RESERVED, p.OPT.SMILEY, p.OPT.SMILEY)
stemmer = SnowballStemmer("english")

def get_dataset_1():
    data1 = pd.read_csv("tweets.txt", delimiter='\t', error_bad_lines=False,
                        names=['UserId', 'CreatedAt', 'Tweet', 'Words'])
    data1 = data1.dropna()
    #data1 = data1.drop(data1.index[data1["UserId"].str.contains("Friday")])

    data1[['UserId']] = data1[['UserId']].astype(int)
    data1['Tweet'] = data1['Tweet'].astype(str)
    data1['CreatedAt'] = pd.to_datetime(data1['CreatedAt'], errors='coerce')
    data1['Words'] = data1['Words'].astype(str)
    data1 = data1.dropna(subset=['CreatedAt'])
    print(data1.shape)

    return data1

def get_dataset_2():

    data2 = pd.read_csv("users.txt", delimiter='\t', error_bad_lines=False,
                        names=['UserId', 'Location', 'Lat', 'Long'])
    data2 = data2.dropna()

    data2['UserId'] = data2['UserId'].astype(int)
    data2['Location'] = data2['Location'].astype(str)
    data2['Lat'] = data2['Lat'].astype(float)
    data2['Long'] = data2['Long'].astype(float)
    data2 = data2.drop_duplicates()
    print(data2.shape)

    return data2

def get_dataset():

    data1 = get_dataset_1()
    data2 = get_dataset_2()

    tweet_dataset = pd.merge(data1, data2, on='UserId', how='inner')
    tweet_dataset = tweet_dataset.dropna()

    print(tweet_dataset.shape)
    return tweet_dataset


def process_tweet(tweet):
    tweet = p.clean(tweet)
    stop = stopwords.words('english') + list(string.punctuation)
    words = [i for i in word_tokenize(tweet.lower()) if i not in stop]
    return words


class TweetSentences(object):
    def __init__(self, fname):
        self.fname = fname

    def __iter__(self):
        for line in open(self.fname, 'r'):
            yield line.split(',')

def process_user_file():
    with open('training_set_users.txt') as f:
        lines = f.readlines()

    dict_loc = {}
    geolocator = Nominatim()

    with open('users.txt') as f1:
        lines1 = f1.readlines()

    for line in lines1:
        x = line.split("\t")
        if len(x) < 2:
            continue

        dict_loc[x[1]] = (x[2], x[3])

    file = open('users.txt', 'a')

    count = 0

    for line in lines:
        if count < 27788:
            count = count + 1
            continue

        x = line.split("\t")
        if len(x) < 2:
            continue

        l = dict_loc.get(x[1])
        if l is None:
            while True:
                try:
                    g = geolocator.geocode(x[1])
                    dict_loc[x[1]] = (str(g.latitude), str(g.longitude))
                    line = x[0] + "\t" + x[1].rstrip() + "\t" + str(g.latitude) + "\t" + str(g.longitude) + "\n"
                # do stuff
                except GeocoderTimedOut as e:
                    print("Exception")
                    continue
                break

            time.sleep(2)
        else:
            line = x[0] + "\t" + x[1].rstrip() + "\t" + l[0] + "\t" + l[1] + "\n"
        file.write(line)
        # users.remove(i)

        count = count + 1
        print(count)

    file.close()



def process_tweet_dataset():
    dataset = get_dataset()

    count = 0

    file = open('tweets.txt', 'w')

    file1 = open('sentences.txt', 'w')

    for index, row in dataset.iterrows():
        tweet = row['Tweet']
        if not tweet:
            continue

        sentence = process_tweet(tweet)
        # tweets_cleaned.append(sentence)
        count = count + 1
        print(count)
        if (len(sentence) == 0):
            continue

        x = ','.join(sentence)

        line = str(row['UserId']) + "\t" + str(row['CreatedAt']) + "\t" + row['Tweet'] + "\t" + x + "\n"
        file.write(line)
        file1.write(x + "\n")

    file.close()
    file1.close()

def cb_sldr_time(attr, old, new):
    global prev_start, prev_end

    val = date_range_slider.value
    print(datetime.fromtimestamp(val[0]/1000), datetime.fromtimestamp(val[1]/1000))
    start_date = datetime.fromtimestamp(val[0]/1000)
    end_date = datetime.fromtimestamp(val[1]/1000)

    start_change = timedelta(0)
    if start_date > prev_start:
        start_change = start_date - prev_start
    elif start_date < prev_start:
        start_change = prev_start - start_date

    end_change = timedelta(0)
    if end_date > prev_end:
        end_change = end_date - prev_end
    elif end_date < prev_end:
        end_change = prev_end - end_date

    print(int(start_change.total_seconds()), int(end_change.total_seconds()))

    if(int(start_change.total_seconds()) < 86400 | int(end_change.total_seconds()) < 86400):
        return
    else:
        prev_end = end_date
        prev_start = start_date

    tweet_subset = tweet_dataset[(tweet_dataset.CreatedAt > start_date) & (tweet_dataset.CreatedAt < end_date)]
    print(tweet_subset.shape)
    src.data['Long'] = tweet_subset['Long'].tolist()
    src.data['Lat'] = tweet_subset['Lat'].tolist()
    src.data['CreatedAt'] = tweet_subset['CreatedAt'].tolist()
    src.data['tweet_text'] = tweet_subset['Tweet'].tolist()
    plt_src.data_source.trigger('data', plt_src.data_source.data,
                                plt_src.data_source.data)
    print(start_date, end_date)

#process_user_file()

def update_visualization():
    global curr_dt
    global flag_play

    tweet_subset = tweet_dataset[tweet_dataset.CreatedAt < curr_dt]
    src.data['Long'] = tweet_subset['Long'].tolist()
    src.data['Lat'] = tweet_subset['Lat'].tolist()
    src.data['CreatedAt'] = tweet_subset['CreatedAt'].tolist()
    plt_src.data_source.trigger('data', plt_src.data_source.data,
                                plt_src.data_source.data)

    x = sldr_rate.value
    curr_dt = curr_dt + timedelta(hours=x)

    print(curr_dt)
    #print(curr_dt.date())

    date_range_slider.start = curr_dt.date()
    date_range_slider.end = curr_dt.date()

    if(curr_dt > end) :
        doc.remove_periodic_callback(update_visualization)
        flag_play = False
        button_play.label = 'Play'


def bt_play_click():
    global flag_play, curr_dt

    if flag_play is False:
        #curr_dt = date_range_slider.start
        doc.add_periodic_callback(update_visualization, 1000)
        button_play.label = 'Pause'
        flag_play = True
    else:
        doc.remove_periodic_callback(update_visualization)
        button_play.label = 'Play'
        flag_play = False

def get_tweets_from_word(symptoms, tweet_dataset):
    catch_words = [x.strip() for x in symptoms.split(',')]
    print(catch_words)
    relevantWords = defaultdict(int)
    relevantWordsSynsets = []
    for catch_word in catch_words:
        synonyms = wordnet.synsets(catch_word)
        lemmas = list(set(chain.from_iterable([catch_word.lemma_names() for catch_word in synonyms])))
        for word in lemmas:
            if word not in relevantWords.keys():
                relevantWords[word] = 0
        relevantWordsSynsets.extend(synonyms)
    relevant_tweet = pd.DataFrame(columns=["Tweet",'Lat',"Long", "CreatedAt"])
    relevant_tweets_only = []
    not_relevant_tweet = []
    # print(tweet_dataset["Words"])
    for index, row in tweet_dataset.iterrows():
        items = row['Words']
        items_tweet = [x.strip() for x in items.split(',')]
        found = 0
        for word in items_tweet:
            if word in relevantWords.keys():
                relevantWords[word] += 1
                relevant_tweets_only.append(row["Tweet"])
                relevant_tweet.loc[len(relevant_tweet)] = [row["Tweet"], row["Lat"], row["Long"],row["CreatedAt"] ]
                found  = 1
                break
        # if found == 0:
        #     not_relevant_tweet.append(original_tweet)
    # Remove words with zero occurrences
    relevantWords = {k: v for k, v in relevantWords.items() if v != 0 and k != catch_word}
    return relevantWords, relevant_tweet, not_relevant_tweet, relevant_tweets_only

def get_trend(relevant_tweets_only):
    date_count = defaultdict(int)
    for tweet in relevant_tweets_only:
        for tweet1 in tweet_dataset.itertuples():
            if tweet == tweet1[3]:
                date = tweet1[4].date()
                date_count[date] += 1
                break
    dates = sorted(date_count)
    counts = []
    for date in dates:
        counts.append(date_count[date])
    return dates, counts

def plot_word_cloud(relevantWords):
    word = []
    font_size = []
    if len(relevantWords.keys()) > 50:
        relevantWords = dict(sorted(relevantWords.items(), key=operator.itemgetter(1), reverse=True)[:50])
    top_count = len(relevantWords.keys())
    x_rand = random.sample(range(1, 100), top_count)
    y_rand = random.sample(range(1, 100), top_count)
    dic = dict(relevantWords)
    s = sum(list(dic.values()))
    if s is 0:
        return
    for key, value in dic.items():
        word.append(key)
        val = (float)(value/s) * 10
        font_size.append("{0:.2f}".format(val) + 'pt')
    source_cloud_1.data['x'] = x_rand
    source_cloud_1.data['y'] = y_rand
    source_cloud_1.data['font_size'] = font_size
    source_cloud_1.data['word'] = word

def plot_trend_graph(dates, counts):
    line1_datasource.data['x'] = dates
    line1_datasource.data['y'] = counts
    line1_datasource.trigger('data', line1_datasource.data, line1_datasource.data)

def bt_compare_click():
    print(search_1.value)
    relevantWords, relevant_tweet, not_relevant_tweet, relevant_tweets_only = get_tweets_from_word(search_1.value, tweet_dataset)
    print(relevantWords)
    dates, counts = get_trend(relevant_tweets_only)
    print(dates)
    print(counts)
    plot_word_cloud(relevantWords)
    plot_trend_graph(dates, counts)


def word_2_vec_computation():

    sentences = TweetSentences('sentences.txt')  # a memory-friendly iterator
    model = Word2Vec(sentences, iter=10)

    model.save('w2v.model')

    #print(model.most_similar('flu', topn=5))


#process_tweet_dataset()
#word_2_vec_computation()


########################get Dataset
tweet_dataset = get_dataset()
tweet_dataset.drop(tweet_dataset[(tweet_dataset.Lat < 20) | (tweet_dataset.Lat > 60)].index, inplace=True)
tweet_dataset.drop(tweet_dataset[(tweet_dataset.Long < -150) | (tweet_dataset.Long > -50)].index, inplace=True)


# relevantWords, relevant_tweet, not_relevant_tweet = get_tweets_for_all_symptoms(tweet_dataset)
relevantWords, relevant_tweet, not_relevant_tweet, relevant_tweets_only = get_tweets_from_word("flu, vomit", tweet_dataset)
dates, counts = get_trend(relevant_tweets_only)

########### Create Visualizations ##################

# Line graph for trend
plot_trend = figure(plot_width=950, plot_height=400, x_axis_type="datetime")
line1 = plot_trend.line(x= dates, y = counts, line_width=2)
line1_datasource = line1.data_source

# Widgets - Search, button
button_go = Button(label="Search Revelant Tweets", width=100, button_type="success")
button_go.on_click(bt_compare_click)

default_search_1 = "flu, vomit"
search_term_1 = default_search_1
search_1 = TextInput(value=default_search_1, title="Enter Symptoms:")


################################
src = ColumnDataSource(data=dict(Lat=[], Long=[], CreatedAt=[],
                                 tweet_text=[]))
# separate latitude and longitude points for the borders
#   of the states.
state_xs = [us_states[code]["lons"] for code in us_states]
state_ys = [us_states[code]["lats"] for code in us_states]

hover2 = HoverTool(tooltips=[
    ("Tweet", "@tweet_text")
])

# init figure
plot_tweet = figure(title="Tweets Plot", toolbar_location="left",
                    plot_width=1100, plot_height=700, tools=[hover2, LassoSelectTool(), ResetTool()])

# Draw state lines
plot_tweet.patches(state_xs, state_ys, fill_alpha=0.0,
    line_color="#884444", line_width=1.5)

plt_src = plot_tweet.circle(x='Long', y='Lat', size=2, color='Red', source=src)
plot_tweet.axis.visible = False
plot_tweet.xgrid.grid_line_color = None
plot_tweet.ygrid.grid_line_color = None

#####################slider, play buttons
sldr_rate = Slider(start=1, end=720, value=24, step=1, title="Rate", width=200)
start = tweet_dataset.CreatedAt.min()
end = tweet_dataset.CreatedAt.max()

prev_start = start
prev_end = end

end_dt = start + timedelta(hours=24)

curr_dt = start

print(start, end, curr_dt)
date_range_slider = DateRangeSlider(title="Date Range: ", start=start.date(), end=end_dt.date(), value=(start.date(), end.date()), step=1000000)
date_range_slider.on_change('value', cb_sldr_time)

button_play = Button(label="Play", width=100, button_type="success")
button_play.on_click(bt_play_click)


# Word clouds for most relevant words
if len(relevantWords.keys()) > 50:
    relevantWords = dict(sorted(relevantWords.items(), key=operator.itemgetter(1), reverse=True)[:50])
top_count = len(relevantWords.keys())
x_rand = random.sample(range(1, 100), top_count)
y_rand = random.sample(range(1, 100), top_count)
word_cloud_stream_1 = figure(x_range=(-20, 120), y_range=(-20, 120),plot_width=400, plot_height=400, tools=[BoxZoomTool(), ResetTool()], title="Revelant Words")
word_cloud_stream_1.toolbar.logo = None
df_stream_1 = pd.DataFrame(pd.np.empty((top_count, 2)) * pd.np.nan, columns=['word', 'weight'])
df_stream_1['word'] = relevantWords.keys()
df_stream_1['weight'] = relevantWords.values()
df_stream_1['font_size'] = df_stream_1['weight'].apply(lambda x: "{0:.2f}".format(x) + 'pt')
df_stream_1['x'] = x_rand
df_stream_1['y'] = y_rand
source_cloud_1 = ColumnDataSource(data = dict(x = x_rand, y= y_rand, word = df_stream_1['word'].tolist(), font_size = df_stream_1['font_size'].tolist()),)
word_cloud_stream_1.text(x='x', y='y', text='word', text_font_size = 'font_size', source=source_cloud_1)
word_cloud_stream_1.axis.visible = False
word_cloud_stream_1.xgrid.grid_line_color = None
word_cloud_stream_1.ygrid.grid_line_color = None

wgt_search = row(widgetbox(search_1), widgetbox(button_go))
wgt_media = row(widgetbox(button_play), widgetbox(sldr_rate), widgetbox(date_range_slider))

doc = curdoc()

layout = layout([wgt_search], [plot_tweet], [wgt_media], [plot_trend, word_cloud_stream_1])

doc.add_root(layout)