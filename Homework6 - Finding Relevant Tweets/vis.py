import string
import time
from datetime import datetime, date
import operator
import simplejson
from collections import defaultdict
import warnings
import random

from bokeh.sampledata import us_states
from bokeh.io import curdoc
from bokeh.models import ColumnDataSource, HoverTool, Dropdown, PreText, Slider, Button, Label, Select, DateRangeSlider
from bokeh.layouts import widgetbox, gridplot, column, layout, row
from bokeh.models import HoverTool, CustomJS,BoxZoomTool, ResetTool, BoxSelectTool, LassoSelectTool, TextInput
from bokeh.plotting import figure
from bokeh.plotting import figure, output_file, show
from bokeh.layouts import widgetbox
from bokeh.models.widgets import TextInput

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

import preprocessor as p

import pandas as pd

import geocoder
from geopy.geocoders import Nominatim, Baidu, Bing, GoogleV3
from geopy.exc import GeocoderTimedOut

# Ignore warnings
warnings.filterwarnings('ignore')

us_states = us_states.data.copy()

default_search_1 = "sick"

del us_states["HI"]
del us_states["AK"]


from gensim.models import Word2Vec
# Setting options for tweet preprocessing - remove URL and emoji
# from preprocessor import s
# p.set_options(p.OPT.URL, p.OPT.EMOJI)
# p.set_options(p.OPT.URL, p.OPT.EMOJI)
# p.set_options(p.OPT.EMOJI,p.OPT.RESERVED, p.OPT.URL, p.OPT.SMILEY)
stemmer = SnowballStemmer("english")

def get_dataset_1():
    data1 = pd.read_csv("tweet_test.txt", delimiter='\t', error_bad_lines=False,
                        names=['UserId', 'TwitterId', 'Tweet', 'CreatedAt'])
    data1 = data1.dropna()
    data1 = data1.drop(data1.index[data1["UserId"].str.contains("Friday")])
    data1[['UserId']] = data1[['UserId']].astype(int)
    data1['Tweet'] = data1['Tweet'].astype(str)
    data1['CreatedAt'] = pd.to_datetime(data1['CreatedAt'], errors='coerce')
    data1 = data1.dropna(subset=['CreatedAt'])
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
    return data2

def get_dataset():
    data1 = get_dataset_1()
    data2 = get_dataset_2()
    tweet_dataset = pd.merge(data1, data2, on='UserId', how='inner')
    tweet_dataset = tweet_dataset.dropna()
    print(tweet_dataset.shape)
    return tweet_dataset

def process_tweet(tweet):
    # tweet = p.clean(tweet)
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
    #data1 = get_dataset_1()
    #userid = list(data1.UserId.unique())
    dict_loc = {}
    geolocator = Nominatim()
    #geolocator= GoogleV3()
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
    tweets = dataset['Tweet'].tolist()
    count = 0
    file = open('abc.txt', 'w')
    for tweet in tweets:
        if not tweet:
            continue
        sentence = process_tweet(tweet)
        # tweets_cleaned.append(sentence)
        count = count + 1
        print(count)
        if (len(sentence) == 0):
            continue
        file.writelines(','.join(sentence) + "\n")

def cb_sldr_time(attr, old, new):
    val = date_range_slider.value
    print(datetime.fromtimestamp(val[0]/1000), datetime.fromtimestamp(val[1]/1000))

tweet_dataset = get_dataset()
# print(tweet_dataset)

tweet_dataset.drop(tweet_dataset[(tweet_dataset.Lat < 20) | (tweet_dataset.Lat > 60)].index, inplace=True)
tweet_dataset.drop(tweet_dataset[(tweet_dataset.Long < -150) | (tweet_dataset.Long > -50)].index, inplace=True)

from itertools import chain
from nltk.corpus import wordnet


def get_tweets_for_all_symptoms(tweet_dataset):
    relevantWords = defaultdict(int)
    # relevantWords = lemmas
    relevantWordsSynsets = []
    for catch_word in ['flu', 'sick', 'fever', 'aches', 'pains', 'fatigue','coughing', 'vomiting' , 'diarrhea']:
        synonyms = wordnet.synsets(catch_word)
        lemmas = list(set(chain.from_iterable([catch_word.lemma_names() for catch_word in synonyms])))
        for word in lemmas:
            if word not in relevantWords.keys():
                relevantWords[word] = 0
        relevantWordsSynsets.extend(synonyms)
        # for i,j in enumerate(wordnet.synsets(catch_word)):
            # hypernyms = j.hypernyms()
            # hyper_lemmas = list(set(chain.from_iterable([word.lemma_names() for word in hypernyms])))
            # for word in hyper_lemmas:
            #     if word not in relevantWords:
            #         relevantWords.append(word)
            # hyponyms = j.hyponyms()
            # hypo_lemmas = list(set(chain.from_iterable([word.lemma_names() for word in hyponyms])))
            # for word in hypo_lemmas:
            #     if word not in relevantWords:
            #         relevantWords.append(word)
            # member_holonyms = j.member_holonyms()
            # member_holonyms_lemmas = list(set(chain.from_iterable([word.lemma_names() for word in member_holonyms])))
            # for word in member_holonyms_lemmas:
            #     if word not in relevantWords:
            #         relevantWords.append(word)
            # part_meronyms = j.part_meronyms()
            # part_meronyms_lemmas = list(set(chain.from_iterable([word.lemma_names() for word in part_meronyms])))
            # for word in part_meronyms_lemmas:
            #     if word not in relevantWords:
            #         relevantWords.append(word)
            # relevantWordsSynsets.extend(hypernyms)
            # relevantWordsSynsets.extend(hyponyms)
            # relevantWordsSynsets.extend(member_holonyms)
            # relevantWordsSynsets.extend(part_meronyms)

    ## similarity
    # for items in tweet_dataset["Tweet"]:
    #         items_tweet = list(items.split())
    #         if set(category_dict['flu']) & set(items_tweet):
    #             flu_tweet.add(items)
    #         else:
    #             wordFromList1 = wordnet.synsets(items_tweet[0])
    #             allsyns1 = relevantWordsSynsets
    #             allsyns2 = set(ss for word in items_tweet for ss in wordnet.synsets(word))
    #             best = max((wordnet.wup_similarity(s1, s2) or 0, s1, s2) for s1, s2 in product(allsyns1, allsyns2))
    #             if(best[0] > 0.9):
    #                 flu_tweet.add(items)
    #             else:
    #                 not_flu_tweet.add(items)

    # relevantWords.append('chills')
    # relevantWords.append('sweats')
    # relevantWords.append("illness")
    # relevantWords.remove("ill")
    # relevantWords.remove("demented")
    # relevantWords.remove("regorge")
    # relevantWords.remove("gruesome")
    # relevantWords.remove("regurgitate")
    # relevantWords.remove("unbalanced")
    # relevantWords.remove("fed_up")
    # relevantWords.remove("disgusted")
    # relevantWords.remove("spew")
    # relevantWords.remove("honk")
    # relevantWords.remove("purge")
    # relevantWords.remove("cat")
    # relevantWords.remove("disturbed")
    # relevantWords.remove("chuck")
    # relevantWords.remove("ghastly")
    # relevantWords.remove("grim")
    # relevantWords.remove("wan")
    # relevantWords.remove("unhinged")
    # relevantWords.remove("mad")
    # relevantWords.remove("grisly")
    # relevantWords.remove("crazy")
    # relevantWords.remove("languish")
    # relevantWords.remove("pine")
    # relevantWords.remove("smart")
    # relevantWords.remove("yearn")
    # relevantWords.remove("yen")
    # relevantWords.remove("hurt")
    # relevantWords.remove("trouble")
    # relevantWords.remove("annoyance")
    # relevantWords.remove("pain_in_the_ass")
    # relevantWords.remove("bother")
    # relevantWords.remove("outwear")
    # relevantWords.remove("wear_upon")
    # relevantWords.remove("wear")
    # relevantWords.remove("jade")
    # relevantWords.remove("wear_down")
    category_dict = {}
    category_dict['flu']  = relevantWords.keys()
    flu_tweet = set()
    not_flu_tweet = set()
    for items in tweet_dataset["Tweet"]:
            original_tweet = items
            items_tweet = set(process_tweet(items))
            if set(category_dict['flu']) & items_tweet:
                commonWords = set.intersection(set(category_dict['flu']), items_tweet)
                for word in list(commonWords):
                    relevantWords[word] += 1
                flu_tweet.add(original_tweet)
            else:
                not_flu_tweet.add(original_tweet)
    # Remove words with zero occurrences
    relevantWords = {k: v for k, v in relevantWords.items() if v != 0}
    return relevantWords, flu_tweet, not_flu_tweet

def get_tweets_from_word(symptoms, tweet_dataset):
    catch_words = [x.strip() for x in symptoms.split(',')]
    print (catch_words)
    print(symptoms)
    relevantWords = defaultdict(int)
    relevantWordsSynsets = []
    for catch_word in catch_words:
        synonyms = wordnet.synsets(catch_word)
        lemmas = list(set(chain.from_iterable([catch_word.lemma_names() for catch_word in synonyms])))
        for word in lemmas:
            if word not in relevantWords.keys():
                relevantWords[word] = 0
        relevantWordsSynsets.extend(synonyms)
        category_dict = {}
        category_dict['relevant'] = relevantWords.keys()
        relevant_tweet = set()
        not_relevant_tweet = set()
        for items in tweet_dataset["Tweet"]:
            original_tweet = items
            items_tweet = set(process_tweet(items))
            if set(category_dict['relevant']) & items_tweet:
                commonWords = set.intersection(set(category_dict['relevant']),items_tweet)
                for word in list(commonWords):
                    relevantWords[word] +=1
                relevant_tweet.add(original_tweet)
            else:
                not_relevant_tweet.add(original_tweet)
    # Remove words with zero occurrences
    relevantWords = {k: v for k, v in relevantWords.items() if v != 0 and k != catch_word}
    return relevantWords, relevant_tweet, not_relevant_tweet

def get_trend(relevant_tweet):
    date_count = defaultdict(int)
    for tweet in relevant_tweet:
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
        val = (float)(value/s) * 100
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
    relevantWords, relevant_tweet, not_relevant_tweet = get_tweets_from_word(search_1.value, tweet_dataset)
    print(relevantWords)
    dates, counts = get_trend(relevant_tweet)
    print(dates)
    print(counts)
    plot_word_cloud(relevantWords)
    plot_trend_graph(dates, counts)


# relevantWords, relevant_tweet, not_relevant_tweet = get_tweets_for_all_symptoms(tweet_dataset)
relevantWords, relevant_tweet, not_relevant_tweet = get_tweets_from_word("flu, vomit", tweet_dataset)
dates, counts = get_trend(relevant_tweet)

########### Create Visualizations ##################

# Line graph for trend
plot_trend = figure(plot_width=950, plot_height=400, x_axis_type="datetime")
# add a line renderer
line1 = plot_trend.line(x= dates, y = counts, line_width=2)
line1_datasource = line1.data_source

# Widgets - Search, button
button_go = Button(label="Search Revelant Tweets", width=100, button_type="success")
button_go.on_click(bt_compare_click)

default_search_1 = "flu, vomit"
search_term_1 = default_search_1
search_1 = TextInput(value=default_search_1, title="Enter Symptoms:")

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
layout = layout([wgt_search], [plot_trend, word_cloud_stream_1])
doc = curdoc()
doc.add_root(layout)
