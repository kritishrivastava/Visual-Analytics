import string
import time
from datetime import datetime, date, timedelta
import random

import operator
from collections import defaultdict
import warnings

from bokeh.sampledata import us_states
from bokeh.io import curdoc
from bokeh.models import ColumnDataSource, PreText, Slider, Button, DateRangeSlider
from bokeh.layouts import widgetbox, column, layout, row
from bokeh.models import HoverTool, BoxZoomTool, ResetTool, LassoSelectTool, TextInput
from bokeh.plotting import figure
from bokeh.models.widgets import RadioGroup

from nltk import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem.snowball import SnowballStemmer

from itertools import chain

import preprocessor as p

import pandas as pd

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

from gensim.models import Word2Vec

us_states = us_states.data.copy()
del us_states["HI"]
del us_states["AK"]

flag_play = False

# Ignore warnings
warnings.filterwarnings('ignore')

p.set_options(p.OPT.EMOJI,p.OPT.RESERVED, p.OPT.SMILEY, p.OPT.SMILEY)
stemmer = SnowballStemmer("english")

#parse from file 1
def get_dataset_1():
    data1 = pd.read_csv("tweets.txt", delimiter='\t', error_bad_lines=False,
                        names=['UserId', 'CreatedAt', 'Tweet', 'Words'])
    data1 = data1.dropna()

    data1[['UserId']] = data1[['UserId']].astype(int)
    data1['Tweet'] = data1['Tweet'].astype(str)
    data1['CreatedAt'] = pd.to_datetime(data1['CreatedAt'], errors='coerce')
    data1['Words'] = data1['Words'].astype(str)
    data1 = data1.dropna(subset=['CreatedAt'])
    print(data1.shape)

    return data1

#get location from file 2
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

#get dataset in required format for visualization
def get_dataset():

    data1 = get_dataset_1()
    data2 = get_dataset_2()

    tweet_dataset = pd.merge(data1, data2, on='UserId', how='inner')
    tweet_dataset = tweet_dataset.dropna()

    print(tweet_dataset.shape)
    return tweet_dataset


#perform preprocessing on the tweet
def process_tweet(tweet):
    tweet = p.clean(tweet)
    stop = stopwords.words('english') + list(string.punctuation)
    words = [i for i in word_tokenize(tweet.lower()) if i not in stop]
    return words

#this is used by the word2vec to read from the file
class TweetSentences(object):
    def __init__(self, fname):
        self.fname = fname

    def __iter__(self):
        for line in open(self.fname, 'r'):
            yield line.split(',')

#this function was used to get location coordinates from city names
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


#function to preprocess each tweet and store list of words corresponding to tweet
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

#callback for the daterange slider
def cb_sldr_time(attr, old, new):
    global prev_start, prev_end

    val = date_range_slider.value
    #print(datetime.fromtimestamp(val[0]/1000), datetime.fromtimestamp(val[1]/1000))
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

    #print(int(start_change.total_seconds()), int(end_change.total_seconds()))

    if(int(start_change.total_seconds()) < 86400 | int(end_change.total_seconds()) < 86400):
        return
    else:
        prev_end = end_date
        prev_start = start_date

    tweet_subset = relevant_tweet[(relevant_tweet.CreatedAt > start_date) & (relevant_tweet.CreatedAt < end_date)]
    print(tweet_subset.shape)
    src.data['Long'] = tweet_subset['Long'].tolist()
    src.data['Lat'] = tweet_subset['Lat'].tolist()
    src.data['CreatedAt'] = tweet_subset['CreatedAt'].tolist()
    src.data['tweet_text'] = tweet_subset['Tweet'].tolist()
    plt_src.data_source.trigger('data', plt_src.data_source.data,
                                plt_src.data_source.data)
    #print(start_date, end_date)
    dates, counts = get_trend(tweet_subset)
    plot_trend_graph(dates, counts)


#callback when the play button is pressed
def update_visualization():
    global curr_dt
    global flag_play, rel_tweet

    tweet_subset = rel_tweet[rel_tweet.CreatedAt < curr_dt]
    src.data['Long'] = tweet_subset['Long'].tolist()
    src.data['Lat'] = tweet_subset['Lat'].tolist()
    src.data['tweet_text'] = tweet_subset['Tweet'].tolist()
    src.data['CreatedAt'] = tweet_subset['CreatedAt'].tolist()
    plt_src.data_source.trigger('data', plt_src.data_source.data,
                                plt_src.data_source.data)

    #print(tweet_subset.shape)
    x = sldr_rate.value
    curr_dt = curr_dt + timedelta(hours=x)

    #print(curr_dt)

    # date_range_slider.start = curr_dt.date()
    # date_range_slider.end = curr_dt.date()

    if(curr_dt > end) :
        curr_dt = start
        doc.remove_periodic_callback(update_visualization)
        flag_play = False
        button_play.label = 'Play'
        date_range_slider.disabled = False

    cur_hours = (curr_dt - start)
    hours = int(cur_hours.total_seconds()/3600)
    dur_text.text = str(hours) + " hr from start"


def plot_scatter(relevant_tweet_df):
    src.data['Long'] = relevant_tweet_df['Long'].tolist()
    src.data['Lat'] = relevant_tweet_df['Lat'].tolist()
    src.data['tweet_text'] = relevant_tweet_df['Tweet'].tolist()
    src.data['CreatedAt'] = relevant_tweet_df['CreatedAt'].tolist()
    plt_src.data_source.trigger('data', plt_src.data_source.data,
                                plt_src.data_source.data)


#callback when play button is pressed
def bt_play_click():
    global flag_play, curr_dt

    if flag_play is False:
        #curr_dt = date_range_slider.start
        doc.add_periodic_callback(update_visualization, 2000)
        button_play.label = 'Pause'
        flag_play = True
        date_range_slider.disabled = True
    else:
        doc.remove_periodic_callback(update_visualization)
        button_play.label = 'Play'
        flag_play = False
        date_range_slider.disabled = False


def get_tweets_from_word(symptoms, tweet_dataset):
    catch_words = [x.strip() for x in symptoms.split(',')]
    #print(catch_words)
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
    #print(relevantWords)
    return relevantWords, relevant_tweet, not_relevant_tweet, relevant_tweets_only

def get_trend(relevant_tweets):
    date_count = defaultdict(int)
    for index, tweet1 in relevant_tweets.iterrows():
        # for index, tweet1 in tweet_dataset.iterrows():
        #     if tweet == tweet1['Tweet']:
        date = tweet1['CreatedAt'].date()
        date_count[date] += 1
        #break
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
    x_rand = random.sample(range(-10, 110), top_count)
    y_rand = random.sample(range(10, 90), top_count)
    dic = dict(relevantWords)
    s = sum(list(dic.values()))
    if s is 0:
        return
    for key, value in dic.items():
        word.append(key)
        if(radio_group.active == 0):
            val = (float)(value/s) * 10
        else:
            val = value * 20

        if(val < 4):
            val = 4
        if(val > 30):
            val = 30
        font_size.append("{0:.2f}".format(val) + 'pt')
    source_cloud_1.data['x'] = x_rand
    source_cloud_1.data['y'] = y_rand
    source_cloud_1.data['font_size'] = font_size
    source_cloud_1.data['word'] = word
    #print(font_size)
    user_text.text = ""

def plot_trend_graph(dates, counts):
    line1_datasource.data['x'] = dates
    line1_datasource.data['y'] = counts
    line1_datasource.trigger('data', line1_datasource.data, line1_datasource.data)

def get_tweet_word2vec(symptoms, tweet_dataset):
    model = Word2Vec.load('w2v.model')
    catch_words = [x.strip() for x in search_1.value.split(',')]
    relevantWords = defaultdict(int)
    for word in catch_words:
        similarWords = model.most_similar(word, topn=20)
        for t in similarWords:
            relevantWords[t[0]] = t[1]
    relevant_tweet = pd.DataFrame(columns=["Tweet", 'Lat', "Long", "CreatedAt"])
    for index, row in tweet_dataset.iterrows():
        items = row['Words']
        items_tweet = [x.strip() for x in items.split(',')]
        for word in items_tweet:
            if word in relevantWords.keys():
                #relevantWords[word] += 1
                #relevant_tweets_only.append(row["Tweet"])
                relevant_tweet.loc[len(relevant_tweet)] = [row["Tweet"], row["Lat"], row["Long"], row["CreatedAt"]]
    # Remove words with zero occurrences
    relevantWords = {k: v for k, v in relevantWords.items() if v != 0}

    #print(relevantWords)
    return relevantWords, relevant_tweet


def update_sliders(df_tweet):
    start = df_tweet.CreatedAt.min()
    end = df_tweet.CreatedAt.max()

    t1 = end - start
    y1 = int((t1.total_seconds() / 3600) / 10)
    y2 = int((t1.total_seconds() / 3600) / 4)

    sldr_rate.end = y2
    sldr_rate.value = y1

    print(y1, y2)
    date_range_slider.start = start.date()
    date_range_slider.end = end.date()
    date_range_slider.value = (start.date(), end.date())

def bt_compare_click():
    user_text.text = "Please wait ....."
    global rel_tweet
    if radio_group.active == 0:
        relevantWords, relevant_tweet, not_relevant_tweet, relevant_tweets_only = get_tweets_from_word(search_1.value,
                                                                                                       tweet_dataset)
        rel_tweet = relevant_tweet.copy(deep=True)
    else:
        # use Word2vec
        relevantWords, relevant_tweet = get_tweet_word2vec(search_1.value, tweet_dataset)
        rel_tweet = relevant_tweet.copy(deep=True)
    dates, counts = get_trend(relevant_tweet)
    plot_word_cloud(relevantWords)
    plot_trend_graph(dates, counts)
    plot_scatter(rel_tweet)
    update_sliders(rel_tweet)

def word_2_vec_computation():

    sentences = TweetSentences('sentences.txt')  # a memory-friendly iterator
    model = Word2Vec(sentences, iter=10)
    #print(model.most_similar('flu', topn=5))

    model.save('w2v.model')


#-----------------function which were used for preprocessing of data
#process_user_file()
#process_tweet_dataset()
#word_2_vec_computation()
#--------------------------------------------------------------------------

########################get Dataset
tweet_dataset = get_dataset()
tweet_dataset.drop(tweet_dataset[(tweet_dataset.Lat < 20) | (tweet_dataset.Lat > 60)].index, inplace=True)
tweet_dataset.drop(tweet_dataset[(tweet_dataset.Long < -150) | (tweet_dataset.Long > -50)].index, inplace=True)


default_search_1 = "flu"
search_term_1 = default_search_1
search_1 = TextInput(value=default_search_1, title="Enter Symptoms:")

# use Word2vec
relevantWords, relevant_tweet = get_tweet_word2vec(default_search_1, tweet_dataset)
rel_tweet = relevant_tweet.copy(deep=True)

dates, counts = get_trend(relevant_tweet)

########### Create Visualizations ##################

# Line graph for trend
plot_trend = figure(title='Trend of Tweets', plot_width=600, plot_height=200, x_axis_type="datetime", tools=[HoverTool(), ResetTool(), BoxZoomTool()])
line1 = plot_trend.line(x= dates, y = counts, line_width=2)
line1_datasource = line1.data_source
plot_trend.xaxis.axis_label = "Date"
plot_trend.yaxis.axis_label = "Relevant Tweet Count"

# Widgets - Search, button
button_go = Button(label="Search Revelant Tweets", button_type="success")
button_go.on_click(bt_compare_click)

rad_text = PreText(text="Choose Word Embedding :")
radio_group = RadioGroup(labels=["WordNet Synset", "Word2Vec"], inline=True, active=1)
user_text = PreText(text="")

################################

src = ColumnDataSource(data=dict(Lat=[], Long=[], CreatedAt=[],
                                 tweet_text=[]))
# separate latitude and longitude points for the borders of the states.
state_xs = [us_states[code]["lons"] for code in us_states]
state_ys = [us_states[code]["lats"] for code in us_states]

hover2 = HoverTool(tooltips=[
    ("Tweet", "@tweet_text")
])

# init figure
plot_tweet = figure(title="Origin of Tweets", toolbar_location="left",
                    plot_width=600, plot_height=400, tools=[hover2, LassoSelectTool(), ResetTool(), BoxZoomTool()])

# Draw state lines
plot_tweet.patches(state_xs, state_ys, fill_alpha=0.0,
    line_color="#884444", line_width=1.5)

plt_src = plot_tweet.circle(x='Long', y='Lat', size=4, color='Red', source=src)
plot_tweet.axis.visible = False
plot_tweet.xgrid.grid_line_color = None
plot_tweet.ygrid.grid_line_color = None

plot_scatter(relevant_tweet)

##################### slider, play buttons
start = relevant_tweet.CreatedAt.min()
end = relevant_tweet.CreatedAt.max()
prev_start = start
prev_end = end
end_dt = start + timedelta(hours=24)
curr_dt = start
#print(start, end, curr_dt)

t1 = end - start
y1 = int((t1.total_seconds()/3600) / 10)
y2 = int((t1.total_seconds()/3600) / 4)
sldr_rate = Slider(start=1, end=y2, value=y1, step=1, title="Rate of change (in hours)")


date_range_slider = DateRangeSlider(title="Date Range: ", start=start.date(), end=end.date(), value=(start.date(), end.date()), step=1000000)
date_range_slider.on_change('value', cb_sldr_time)

button_play = Button(label="Play", button_type="success")
button_play.on_click(bt_play_click)

cur_hours = (curr_dt - start)/3600
hours = int(cur_hours.total_seconds()/3600)
dur_text = PreText(text=str(hours) + " hr from start", height=50)


######################## Word clouds for most relevant words
if len(relevantWords.keys()) > 40:
    relevantWords = dict(sorted(relevantWords.items(), key=operator.itemgetter(1), reverse=True)[:40])
top_count = len(relevantWords.keys())
x_rand = random.sample(range(-10, 110), top_count)
y_rand = random.sample(range(20, 80), top_count)
word_cloud_stream_1 = figure(x_range=(-20, 120), y_range=(0, 100),plot_width=600, plot_height=300, tools=[BoxZoomTool(), ResetTool()], title="Similar Words")
word_cloud_stream_1.toolbar.logo = None
word_cloud_stream_1.axis.visible = False
word_cloud_stream_1.xgrid.grid_line_color = None
word_cloud_stream_1.ygrid.grid_line_color = None

source_cloud_1 = ColumnDataSource(data = dict(x = [], y= [], word = [], font_size = []))
word_cloud_stream_1.text(x='x', y='y', text='word', text_font_size='font_size', source=source_cloud_1)
plot_word_cloud(relevantWords)


wgt_select = column(widgetbox(rad_text), widgetbox(radio_group))
wgt_but = column(widgetbox(button_go), widgetbox(user_text))
wgt_search = row(children=[widgetbox(search_1)])
wgt_media_1 = row(widgetbox(button_play), widgetbox(dur_text))
wgt_media_2 = row(widgetbox(sldr_rate), widgetbox(date_range_slider))

doc = curdoc()

layout = layout([wgt_search, wgt_select, wgt_but], [[[plot_tweet], [wgt_media_1], [wgt_media_2]], [[plot_trend],[word_cloud_stream_1]]])

doc.add_root(layout)