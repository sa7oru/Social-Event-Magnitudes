'''Thanks to http://charuaggarwal.net/sdm2012-camera.pdf'''

# For Computation

import os, sys, email, re, random, math, time, datetime
import matplotlib.pyplot as plt
import pandas as pd
import math
from datetime import timedelta
from StringIO import StringIO
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from collections import Counter, deque

# For natural language processing

import string
import preprocessor as twprep
from nltk.tokenize.regexp import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

# Functions

def striptimedelta(interval):

    timedelta_string = str(interval)
    formatted_timedelta = "_".join(timedelta_string.split(":"))

    return formatted_timedelta

def Sim_Structure (social_stream_entity, cluster):

    result = 0.0
    views = social_stream_entity.viewers
    total_views = cluster.total_viewers

    for i in social_stream_entity.all_nodes:
        if cluster.nodes_summary.has_key(i):
            result += cluster.nodes_summary.get(i)

    a = pow(len(social_stream_entity.all_nodes)+views, 1/2)
    b = total_views + sum(cluster.nodes_summary.itervalues())

    if b == 0:
        pass
    else:
        result = result/(a*b)

    return result

def Sim_Content (social_stream_entity, cluster):

    if social_stream_entity.content == " " or cluster.words_summary == "":

        similarity = 0.0

    else:
        docs = [social_stream_entity.content, cluster.words_summary]

        tfidf = TfidfVectorizer(stop_words="english")

        try:

            temp = tfidf.fit_transform(docs)
            doc_social_stream_entity = temp[0].todense()
            doc_cluster = temp[1].todense()
            similarity = cos_sim(doc_social_stream_entity, doc_cluster)[0][0]

        except:

            similarity = 0.0

    return similarity

def get_sim (rh, clusters, gamma):

    sim = []

    for index, entity in enumerate(clusters):
        sim.append((index, gamma*Sim_Content(rh, entity) + (1-gamma)*Sim_Structure(rh, entity)))

    return sim

def update_clusters_return_sim_loc (sim, mean, std, clusters, soc_obj):

    max_sim_loc = max(sim, key = lambda tup: tup[1])
    max_sim = max_sim_loc[1]
    max_loc = max_sim_loc[0]
    returned_sim = 0
    returned_loc = 0

    if max_sim > (mean - 3*std):
        clusters[max_loc].add_tags(soc_obj)
        clusters[max_loc].update_nodes_summary(soc_obj)
        clusters[max_loc].update_words_summary(soc_obj)
        returned_sim = max_sim
        returned_loc = max_sim_loc[0]
    else:
        close_sim_loc = min(sim, key = lambda tup: tup[1]-(mean - 3*std))
        close_loc = close_sim_loc[0]
        close_sim = close_sim_loc[1]
        clusters[close_loc].add_tags(soc_obj)
        clusters[close_loc].update_nodes_summary(soc_obj)
        clusters[close_loc].update_words_summary(soc_obj)
        returned_sim = close_sim
        returned_loc = close_loc

    return (returned_sim, returned_loc)

def update_statistics (the_returned, similarities, squared_deviations, count, mean, std):

    count += 1
    similarities = similarities + the_returned[0]
    first_moment = similarities/count
    squared_deviations = squared_deviations + pow((the_returned[0] - first_moment), 2)
    second_moment = squared_deviations/count
    mean = first_moment
    std = pow((second_moment - pow(mean, 2)), 0.5)

    if np.isnan(std):
        std = 0
    else:
        pass

    return similarities, squared_deviations, count, mean, std

def sum_time (timeList):

    total = timedelta()

    for i in timeList:

        total += i

    return total

def check_event_specific (subject):

    result = 0
    labels = ['charlottesville']
    try:
        if set(subject.split()).intersection(set(labels)) != set([]):
            result = 1
    except:
        print subject.partition(" ")
        raise

    return result

def get_event_sig (clusters):

    event_sig = []

    for a in clusters:
        if a.number_of_members > 0:
            event_sig.append(round(a.ground_truth/float(a.number_of_members), 3))
        else:
            event_sig.append(0)

    return event_sig

def get_horizon_sig (h_loc, number_of_clusters):

    horizon_sig = [0 for x in range(number_of_clusters)]

    length_recent_history = len(h_loc)

    for hl in h_loc:
        horizon_sig[hl] += 1/float(length_recent_history)

    return horizon_sig

def clean(text):
    stop = set(stopwords.words('english'))

    stop.update(("to","ya","pm","cc","subject","http","enron","com","from","sent","thank","please","hou","ect","fyi", "image"
                "a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","z","x","y"))

    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()

    if text == None:
        pass
    else:
        text = re.sub(r'\S*@\S*\s', '', text.rstrip())
        text = re.sub(r'Original Message', '', text)
        text = re.sub(r'\S*\.\S\S\S', '', text)
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        stop_free = " ".join([i for i in text.lower().split() if ((i not in stop) and (not i.isdigit()))])
        punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
        normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())

    return str(normalized)

# Objects

class social_stream_entity:

    def __init__ (self, sender, recipients, viewers, content, hashtags, date_time, delta_time, happened):

        if content == "":
            self.content = " "
        else:
            self.content = content

        self.happened = happened
        self.viewers = viewers
        self.sender = sender
        self.tag = " "
        self.recipients = set()
        self.predicted = 0
        self.date_time = date_time
        self.delta_time = delta_time

        if recipients != {"na"}:
            try:
                for i in recipients:
                    self.recipients.add(i)
            except:
                print "???: %s \n" %(recipients)

        labels = ['charlottesville']

        for i in labels:
            if i in hashtags.split():
                if self.tag != " ":
                    self.tag = random.choice([i, self.tag])
                else:
                    self.tag = i

        self.all_nodes = self.recipients.union(self.sender)

class cluster:

    def __init__(self, social_stream_entity):

        self.tags = []
        self.last_update = time.time()
        self.nodes_summary = dict()

        if social_stream_entity is None:
            self.number_of_members = 0
            self.words_summary = ""
            self.ground_truth = 0
            self.total_viewers = 0
        else:
            self.number_of_members = 1
            self.words_summary = social_stream_entity.content
            self.tags.append((social_stream_entity.date_time, social_stream_entity.tag))
            self.ground_truth = social_stream_entity.happened
            self.total_viewers = social_stream_entity.viewers
            for i in social_stream_entity.all_nodes:
                self.nodes_summary[i] = 1

    def add_tags(self, new_social_stream_entity):

        self.number_of_members += 1
        self.tags.append((new_social_stream_entity.date_time, new_social_stream_entity.tag))
        self.last_update = time.time()
        self.ground_truth += new_social_stream_entity.happened
        self.total_viewers += new_social_stream_entity.viewers

    def update_nodes_summary(self, new_social_stream_entity):

        if new_social_stream_entity != None:
            for i in new_social_stream_entity.all_nodes:
                if self.nodes_summary.has_key(i) == False:
                    self.nodes_summary[i] = 1
                else:
                    self.nodes_summary[i] += 1
        else:
            pass

    def update_words_summary(self, new_social_stream_entity):

        if new_social_stream_entity.content == " " or new_social_stream_entity.content == "":
            pass
        else:
            self.words_summary = " ".join([self.words_summary, new_social_stream_entity.content])

# Main

def main():

    # User's input Start
    '''
    input_data = <User Input: file location, file extention: .csv>
    the_chunk = <User Input: number of data points >
    number_of_clusters = 750
    interval = timedelta(minutes = 1, seconds = 0)
    minimum_records = <User Input: any number bigger 3>
    precision_recall = <User Input: file location, file extention: .txt >
    the_result = <User Input: file location, file extention: .png>
    the_graph = <User Input: file location, file extention: .xlsx>
    '''
    # User's Input End

    mean = 0
    std = 0
    gamma = 0.5
    similarities = 0
    squared_deviations = 0
    count = 0
    history = deque()
    gap = timedelta(seconds = 0)
    time_zero = timedelta(hours = 0)
    filename_interval =  striptimedelta(interval)
    social_objects = []

    print "Testing SE Twitter with data points: %d, time: %s, clusters: %s, min_records: %d" %(the_chunk, interval, number_of_clusters, minimum_records)

    chunk = pd.read_csv(input_data, chunksize = the_chunk)
    data = next(chunk)
    data = data.fillna("na")
    start_time = time.time()

    # Data preprocessing

    twprep.set_options(twprep.OPT.URL, twprep.OPT.EMOJI, twprep.OPT.SMILEY, twprep.OPT.NUMBER)
    data['full_text'] = data['full_text'].map(lambda x: clean(twprep.clean(x)))
    data['hashtags'] = data['hashtags'].map(clean)
    data['screen_name'] = data['screen_name'].map(lambda x: {x.lower()})
    data['in_reply_to_screen_name'] = data['in_reply_to_screen_name'].map(lambda x: {x.lower()})
    data['created_at'] = pd.to_datetime(data['created_at'], infer_datetime_format=False)
    data['viewers'] = data.apply(lambda x: x['friends_count'] + x['followers_count'], axis = 1)
    data = data.set_index('created_at')
    data = data.sort_index()
    data['created_at_copy'] = data.index
    data['delta_time'] = data['created_at_copy'].subtract(data['created_at_copy'].shift()).fillna(0)
    data = data.dropna()

    analysis_data = data[['screen_name', 'full_text', 'viewers', 'created_at_copy', 'delta_time', 'in_reply_to_screen_name', 'hashtags']].dropna().copy()

    # Setting ground truth

    analysis_data = analysis_data.dropna()
    analysis_data["Happened"] = 0
    gangan = 0

    for index, row in analysis_data.iterrows():
        gangan += 1
        print "Marking ground truth, %d remains" %(the_chunk - gangan)
        history.append((row['delta_time'], check_event_specific(row['hashtags'])))

        h = timedelta(minutes = 0)
        r = 0

        for rh in reversed(history):
            h += rh[0]
            r += rh[1]
            if interval == h and r >= minimum_records:
                analysis_data.at[index, 'Happened'] = 1
                break

        while sum_time([x[0] for x in history]) > interval:
            history.popleft()

    combined = zip(analysis_data['screen_name'], analysis_data['in_reply_to_screen_name'], analysis_data['viewers'], analysis_data['full_text'], analysis_data['hashtags'], analysis_data['created_at_copy'],analysis_data['delta_time'], analysis_data['Happened'])

    print "Creating clusters and social objects"

    clusters = [cluster(None) for i in range(number_of_clusters)]

    for sender, recipients, viewers, words, hashtags, date_time, delta_time, happened in combined:
        social_objects.append(social_stream_entity(sender, recipients, viewers, words, hashtags, date_time, delta_time, happened))

    alarms = []
    h_loc = []
    recent_history_loc = deque()
    supervised_start = time.time()

    for index, so in enumerate(social_objects):

        print "Detecting, remaining: %d \n" %(the_chunk - index - 1)
        print "\tMean: %f, Standard Dev: %f" %(mean, std)

        h = timedelta(hours = 0)
        loc = []
        one_loop_time = time.time()
        the_returned = update_clusters_return_sim_loc(get_sim(so, clusters, gamma), mean, std, clusters, so)
        print "\tSim Avg: %f, Assigned to: %d \n" %(the_returned[0], the_returned[1])
        print "\tTook: %f seconds\n" %(time.time() - one_loop_time)
        recent_history_loc.append((so.delta_time, the_returned[1]))
        similarities, squared_deviations, count, mean, std = update_statistics(the_returned, similarities, squared_deviations, count, mean, std)
        horizon_difference = sum_time([x[0] for x in recent_history_loc]) - interval

        for rr in reversed(recent_history_loc):
            h += rr[0]
            loc.append(rr[1])

            if  interval == h:

                horizon_sig = get_horizon_sig(loc, number_of_clusters)
                event_sig = get_event_sig(clusters)
                alarms.append((so, np.dot(np.array(event_sig), np.array(horizon_sig))))
                break

        while sum_time([x[0] for x in recent_history_loc]) > interval:
            recent_history_loc.popleft()

    run_time = (time.time() - supervised_start)/60
    print " "
    print "Testing: SE Twitter with interval %s hours" %interval
    print "Took: %f minutes" %(run_time)
    print "Speed: %f data points per minute" %(the_chunk/run_time)
    print "Number of non-empty cluster: %d" %(len([x for x in clusters if x.number_of_members != 0]))

    output_alarms = [ (x[0].happened, x[1]) for x in alarms]
    with open(the_result, 'a') as sr:
        sr.write("%s\n" %output_alarms)
        sr.write("Took: %f minutes\n" %(run_time))
        sr.write("Speed: %f data points per minute\n" %(the_chunk/run_time))
        sr.write("Number of non-empty cluster: %d" %(len([x for x in clusters if x.number_of_members != 0])))

    x_axis_recall = []
    y_axis_precision = []

    for t in np.arange(1, 0.01, -0.005):

        predicted = 0
        relevant_retrieved = 0
        relevant = 0

        for sa in alarms:

            relevant += sa[0].happened

            if sa[1] > round(t,4):
                sa[0].predicted = 1
                predicted += 1

                if (sa[0].predicted == 1) and (sa[0].happened == 1):
                    relevant_retrieved += 1
            else:
                sa[0].predicted = 0

        x_axis_recall.append(relevant_retrieved/float(relevant))

        if float(predicted) > 0:
            y_axis_precision.append(relevant_retrieved/float(predicted))
        else:
            y_axis_precision.append(0)

    with open(precision_recall, 'a') as pr:
        pr.write("Recall: %s \n" %x_axis_recall)
        pr.write("Precision: %s \n" %y_axis_precision)

    plt.plot(x_axis_recall, y_axis_precision, '-s')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig(the_graph)

if __name__ == '__main__':
    main()
    print "The program is done.\n"
