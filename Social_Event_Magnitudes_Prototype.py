'''Built according to
1. https://www.kdd.org/kdd2016/papers/files/adf0510-nikolaevA.pdf
2. https://arxiv.org/pdf/1306.5550.pdf
'''

# For Computation

import scipy
import os, sys, email, re, random, time, math, datetime
from datetime import timedelta
import numpy as np
import pandas as pd
from scipy.linalg import eigh
import anytree
from anytree import Node, RenderTree, AsciiStyle, PreOrderIter, LevelOrderGroupIter, Walker
from collections import Counter, deque
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

# For plotting

import matplotlib.pyplot as plt

# For natural language processing

import preprocessor as twprep
from nltk.tokenize.regexp import RegexpTokenizer
from nltk import word_tokenize, pos_tag
from subprocess import check_output
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string

# Functions

def striptimedelta(interval):

    timedelta_string = str(interval)
    timedelta_string = timedelta_string.split(":")
    formatted_timedelta = "_" + str(timedelta_string[0]) + "h_" + str(timedelta_string[1]) + "m_" + str(timedelta_string[2]) + "s"

    return formatted_timedelta

def clean(text):

    stop = set(stopwords.words('english'))
    stop.update(("to","ya","pm", "html","cc", "image","http","else","enron","com","from","sent","thank","please","hou","ect","fyi", "image"
                "a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","z","x","y"))

    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()

    if text is None or text == "" or text == " ":
        normalized = " "
    else:
        text = re.sub(r'\S*@\S*\s', '', text.rstrip())
        text = re.sub(r'Original Message', '', text)
        text = re.sub(r'\S*\.\S\S\S', '', text)
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        stop_free = " ".join([i for i in text.lower().split() if ((i not in stop) and (not i.isdigit()))])
        punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
        normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())

    return str(normalized)

def sum_PowerSeries(parameter, number_of_terms):
    initial = 1.0
    for i in range(number_of_terms-1):
        initial += parameter**i
    return initial

def check_event_specific (hashtag):

    result = 0

    labels = ['charlottesville']

    try:
        if set(hashtag.split()).intersection(set(labels)) != set([]):
            result = 1
    except:
        print hashtag.partition(" ")
        raise

    return result

def EC_calculate(children, parameter):

    if children == ():
        ec = 0

    else:

        temp = 0
        ec = len(children)*pow(parameter, len(children[0].path) - children[0].depth)/sum_PowerSeries(1, children[0].depth)
        for i in children:

            ec += EC_calculate(i.children, parameter)
            temp = ec

    return ec

def sum_time (timeList):

    total = timedelta()

    for i in timeList:

        total += i

    return total

def is_similar (message, group_content, cutoff):

    result = False

    if message == " " or message == "" or len(group_content) == 0:

        result = False

    else:

        content = " ".join(list(group_content))

        docs = [message, content]

        tfidf = TfidfVectorizer(stop_words= "english")

        try:

            temp = tfidf.fit_transform(docs)
            docs_1 = temp[0].todense()
            docs_2 = temp[1].todense()

            if cos_sim(docs_1, docs_2)[0][0] >= cutoff:

                result = True

            else:

                result = False

        except:

            result = False

    return result

def NB_Spectral (co_occurrence):

    sum_per_row = np.sum(co_occurrence, axis = 1)
    EmptyMatrix = scipy.sparse.csr_matrix(co_occurrence.shape)
    Identity = scipy.sparse.identity(n = co_occurrence.shape[0])
    Diagnol = scipy.sparse.spdiags(sum_per_row.A1, 0, co_occurrence.shape[0], co_occurrence.shape[0])

    NB_upper_half = scipy.sparse.hstack([co_occurrence, (Identity-Diagnol)])
    NB_lower_half = scipy.sparse.hstack([Identity, EmptyMatrix])
    NB = scipy.sparse.vstack([NB_upper_half, NB_lower_half])

    print "\n\tSize of NB Spectral: %d" %NB.shape[0]
    nb_cal_time = time.time()
    eigenvalues = scipy.sparse.linalg.eigsh(NB, which = 'BE', k = 2, return_eigenvectors = False)
    print "\tNB Spectral took: %f seconds \n" %(time.time() - nb_cal_time)
    return eigenvalues

def event_over_norm (potential_events):

    max_event = max(potential_events, key = lambda x:x > 0 )
    norm = np.linalg.norm(potential_events)
    ratio = 0

    if norm > 0:
        ratio = max_event/float(norm)
    else:
        pass

    return norm

def product_eval1_eval2(eigenvalues, engaging_groups, tree_para, sqrt_avg_degrees):

    eval1 = sorted(eigenvalues, reverse = True)[0]

    if eval1 <= sqrt_avg_degrees:
        eval1 = 0

    speakers_to_group_matrix = get_speakers_to_groups_matrix (engaging_groups, tree_para)
    print "\tSize of Engaging Matrix: %s" %(speakers_to_group_matrix.shape, )

    engaging_matrix = speakers_to_group_matrix.T*speakers_to_group_matrix

    eng_eval = eigh(engaging_matrix.todense(), eigvals_only = True)
    eval2 = sorted(eng_eval, reverse = True)[0]

    return eval1, eval2, eval1*eval2

def find_true_events (potential_events):

    true_events = potential_events

    true_events = sorted(zip(true_events, range(len(true_events))), key = lambda tup: tup[0], reverse = True)

    return true_events

def get_term_to_speakers_matrix (speeches_all_records):

    speech = pd.DataFrame.from_dict(speeches_all_records )
    words = speech.columns.get_values()
    speech = speech.drop_duplicates(subset = {"the_speaker"}, keep='first')
    speech = speech.set_index('the_speaker')
    speech = speech.sort_index()
    speech = speech.fillna(0).astype(int)
    term_to_speakers_matrix = scipy.sparse.csr_matrix(speech.values).T

    return term_to_speakers_matrix, words.tolist()

def get_speakers_to_groups_matrix (engaging_groups, tree_para):

    temp = []

    for index, group in enumerate(engaging_groups):
        group.get_engagement_capacities(tree_para)
        temp.append(group.capacities)

    group_to_person = pd.DataFrame.from_dict(temp).fillna(0)

    group_to_person = group_to_person[sorted(group_to_person.columns)]

    speakers_to_groups_matrix = scipy.sparse.csr_matrix(group_to_person).T

    return speakers_to_groups_matrix

def find_max_potential_events (eigenvalues, eigenvectors, fused_speeches_cooccur, engaging_groups, tree_para, top_potentials):

    A, _ = get_term_to_speakers_matrix(fused_speeches_cooccur)
    sizeA = np.shape(A)
    number_of_words = sizeA[0]

    B = get_speakers_to_groups_matrix(engaging_groups, tree_para)
    sizeB = np.shape(B)

    power_word_locations = []
    potential_events = 0

    max_eigen_pair = sorted(zip(eigenvalues, eigenvectors), key = lambda tup: tup[0], reverse = True)[0]
    max_eigenvalue = max_eigen_pair[0]

    max_eigenvector = np.array([x if x > 0 else 0 for x in max_eigen_pair[1][:number_of_words]])
    sorted_vec = sorted(zip(max_eigenvector, range(len(max_eigenvector))), key = lambda tup: tup[0] > 0, reverse = True)

    for i in sorted_vec[:top_potentials]:
        power_word_locations.append(i[1])

    if sizeA[1] != sizeB[0]:

        print "term to speakers: %d, speakers to groups: %d \n" %(sizeA[1], sizeB[0])

        speech_shit = pd.DataFrame.from_dict(fused_speeches_cooccur)
        speech_shit = speech_shit.drop_duplicates(subset = {"the_speaker"}, keep='first')
        speech_shit = speech_shit.set_index('the_speaker')

        someshit = []

        for p, g in enumerate(engaging_groups):
            g.get_engagement_capacities(tree_para)
            someshit.append(g.capacities)

        gtop = pd.DataFrame.from_dict(someshit)

        print "Speakers from term-to-speaker: %s \n" %sorted(speech_shit.index)
        print "Speakers from people-to-groups: %s \n" %sorted(gtop.columns)
        print "WHO THE FUCK: %s \n" %set(speech_shit.index).symmetric_difference(set(gtop.columns))

    else:

        potential_groups = A*B
        potential_events = max_eigenvector.T*potential_groups

    if len(power_word_locations) == 0:
        potential_events = 0

    return power_word_locations, potential_events

def prepare_co_occurrence (list_of_docs):

    # rows means (across) documents
    # columns means (across) terms
    # values means 1 for (rows, columns) if the document has that term

    rows = []
    columns = []
    values = []
    temp = []

    for i in list_of_docs:
        for w in i["the_words"]:
            if w in temp:
                pass
            else:
                temp.append(w)

    sorted_temp = sorted(temp)

    voc2id = dict(zip(sorted_temp, range(len(sorted_temp))))

    for index, doc in  enumerate(list_of_docs):

        for word in doc["the_words"]:
            if voc2id.has_key(word):
                columns.append(index)
                rows.append(voc2id[word])
                values.append(1)

    return rows, columns, values, len(voc2id), len(list_of_docs)

def fuse_speeches_cooccur (speeches_all_records, cooccur_all_records):

    result = []

    for s, c in zip(speeches_all_records, cooccur_all_records):

        personal_record = dict()
        personal_record["the_speaker"] = s[0]

        for w in c["the_words"]:
            personal_record[w] = 1

        result.append(personal_record)

    return result

# Objects

class engaging_group:

    def __init__(self, content, members, date_time, gap, interval, messageID, viewers):

        time_zero = timedelta(hours = 0)
        self.content = set(content.split())

        self.viewers = viewers
        self.recorded_speakers = [{"the_speaker":members[0], "the_time": date_time, "the_id": messageID, "the_viewers": viewers}]
        self.messageIDs = []
        self.messageIDs.append(messageID)
        self.tree = []
        self.root = Node(self.recorded_speakers[0]["the_speaker"])
        self.tree.append(self.root)
        self.previous_entry_time = date_time
        self.capacities = dict() # Here we just initialize to zero; it won't be calculated until get_engagement_capacities is called

        for i in self.tree:
            self.capacities[i.name] = 0.0

    def update_tree(self, date_time, gap, interval, new_mem, new_viewers, messageID):

        self.recorded_speakers.append({"the_speaker": new_mem[0], "the_time": date_time, "the_id": messageID, "the_viewers": new_viewers})
        self.previous_entry_time = date_time
        self.viewers += new_viewers

        for i in new_mem:
            if self.capacities.has_key(i):
                pass
            else:
                self.capacities[i] = 0.0

        if interval == 0:
            for i in new_mem:
                self.tree.append(Node(i, parent = self.tree[-1]))

        else:

            if date_time - self.previous_entry_time - interval <= gap:

                leveled_tree = [[node.name for node in children] for children in LevelOrderGroupIter(self.tree[0])]

                temp = []

                for j in self.tree:
                    if j.name in leveled_tree[-1]:
                        temp.append(j)

                # There are 2 list needed for randomly updating the last level of the tree, one from new_mem,
                # and the other one from the latest level of the tree (temp).

                while len(new_mem)>0:
                    index_new_mem = random.randrange(len(new_mem))
                    index_temp = random.randrange(len(temp))
                    tree_element_new_mem = new_mem[index_new_mem]
                    tree_element_temp = temp[index_temp]
                    del new_mem[index_new_mem]
                    self.tree.append(Node(tree_element_new_mem, parent = tree_element_temp))
            else:

                for i in new_mem:
                    self.tree.append(Node(i, parent = self.tree[-1]))

    def remove_old(self, now, interval):

        self.recorded_speakers = [x for x in self.recorded_speakers if now - x.get("the_time") <= interval]

        if len(self.recorded_speakers) == 0:

            pass

        else:

            self.viewers = sum([x.get("the_viewers") for x in self.recorded_speakers])
            self.messageIDs = [x.get("the_id") for x in self.recorded_speakers]
            self.capacities = dict()
            self.tree = []
            self.root = Node(self.recorded_speakers[0]["the_speaker"])
            self.tree.append(self.root)
            temp_time = self.recorded_speakers[0]["the_time"]

            for i in self.recorded_speakers[1:]:

                if i.get("the_time") - temp_time > interval:
                    self.tree.append(Node(i.get("the_speaker"), parent = self.tree[-1]))
                    temp_time = i.get("the_time")

                else:

                    leveled_tree = [[node.name for node in children] for children in LevelOrderGroupIter(self.tree[0])]
                    temp = []

                    if len(leveled_tree) > 1:

                        for j in self.tree:

                            if j.name in leveled_tree[-2]:
                                temp.append(j) # get the nodes from the last level

                        temp_index = random.randrange(len(temp))
                        temp_tree_node = temp[temp_index]
                        self.tree.append(Node(i.get("the_speaker"), parent = temp_tree_node))

                    else:

                        for j in self.tree:

                            if j.name in leveled_tree[-1]:
                                temp.append(j)

                        temp_index = random.randrange(len(temp))
                        temp_tree_node = temp[temp_index]
                        self.tree.append(Node(i.get("the_speaker"), parent = temp_tree_node))

            for i in self.tree:
                self.capacities[i.name] = 0.0

    def get_engagement_capacities(self, parameter):
        future_influence = 1

        if self.viewers > 1:
            future_influence = 1/math.log(self.viewers)
        else:
            pass

        for i in self.tree:
            if i.descendants != ():
                self.capacities[i.name] += EC_calculate(i.children, parameter*future_influence)

class speech_records_materials:

    def __init__ (self):

        self.all_records = []

    def update_personal_record (self, new_speaker, new_content, date_time, interval):

        self.all_records.append((new_speaker, date_time))

    def remove_old (self, now, interval):

        if len(self.all_records) > 0:
            try:
                self.all_records = [x for x in self.all_records if now - x[1] <= interval]
            except:
                print "now: %s" %now
                print "All records: %s" %self.all_records
                raise

class co_occurrence_materials:

    def __init__(self, content):

        self.recorded_words = []

    def update_recorded_words(self, new_doc, date_time):

        terms = new_doc.split()

        temp = dict()
        new_terms = []

        for term in terms:
            if term in new_terms:
                pass
            else:
                new_terms.append(term)

        temp["the_words"] = new_terms
        temp["the_time"] = date_time
        self.recorded_words.append(temp)

    def remove_old(self, now, interval):

        if len(self.recorded_words) > 0:
            self.recorded_words = [x for x in self.recorded_words if now - x.get("the_time") <= interval]
        else:
            pass

# Main

def main():

    # User's input Start
    '''
    input_data = <User Input: file location, file extention: .csv>
    the_chunk = <User Input: number of data points >
    cutoff = <User Input: any value between 0.1 and 1>
    tree_para = <User Input: any value between 0.1 and 1>
    minimum_records = <User Input: any number bigger 3>
    gap = timedelta(seconds = 0)
    interval = timedelta(minutes = 0, seconds = 30)
    precision_recall = <User Input: file location, file extention: .txt >
    the_graph = <User Input: file location, file extention: .png>
    raw_result = <User Input: file location, file extention: .xlsx>
    '''
    # User's Input End

    history = deque()
    supervised_pairs = []
    recent_history = deque()
    time_zero = timedelta(minutes = 0)
    cum_time = timedelta(hours = 0)
    x_axis_time = []
    y_axis_engaging = []
    background = []
    groups_engaging = []
    engaging_groups = []
    cooccur = co_occurrence_materials(" ")
    speeches = speech_records_materials()
    filename_interval = striptimedelta(interval)

    print "Testing My Algo Twitter with data points: %d, time: %s, cutoff: %s, min_records: %d" %(the_chunk, interval, cutoff, minimum_records)

    chunk = pd.read_csv(input_data, chunksize = the_chunk)
    data = next(chunk)
    data = data.fillna("na")
    start_time = time.time()

    # Data preprocessing

    twprep.set_options(twprep.OPT.URL, twprep.OPT.EMOJI, twprep.OPT.NUMBER, twprep.OPT.SMILEY)
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
    print data.shape
    analysis_data = data[['id', 'screen_name', 'in_reply_to_screen_name', 'viewers', 'delta_time', 'created_at_copy', 'full_text', 'hashtags']].dropna().copy()
    for_gaussian_event_count = []

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

            if interval - h == time_zero and  r >= minimum_records:

                analysis_data.at[index, 'Happened'] = 1
                break

        while sum_time([x[0] for x in history]) > interval:
            history.popleft()

    combined = zip(analysis_data['id'], analysis_data['screen_name'], analysis_data['in_reply_to_screen_name'], analysis_data['delta_time'], analysis_data['created_at_copy'], analysis_data['full_text'], analysis_data['Happened'], analysis_data['viewers'])
    dynamic_del_count = 0
    count = 0
    supervised_start = time.time()
    break_point = len(combined)
    break_point_count = 0

    for messageID, speaker, receiver, delta_time, date_time, content, happened, viewers in combined:

        break_point_count += 1

        print "Remaining: %d \n" %(break_point - break_point_count)

        if break_point_count == break_point:
            print "Ran through"
            break

        recent_history.append(delta_time)

        cum_time = sum_time([x for x in recent_history])
        cooccur.update_recorded_words(content, date_time)
        same_speaker = next(iter(speaker))
        speeches.update_personal_record(same_speaker, content, date_time, interval)

        if len(engaging_groups) == 0:

            engaging_groups.append(engaging_group(content, [same_speaker], date_time, gap, interval, messageID, viewers))

        else:

            number_of_groups = len(engaging_groups)

            appendix_counter = 0

            for group in engaging_groups[:number_of_groups]:

                appendix_counter += 1 # record number of comparisons

                # Grouped by content or receiver

                if len(receiver.intersection(set([t.name for t in group.tree]))) >0 or is_similar(content, group.content, cutoff) :
                    group.update_tree(date_time, gap, interval, [same_speaker], viewers, messageID)
                    group.content.update(set(content.split()))
                    break

                else:
                    if appendix_counter < number_of_groups:
                        pass
                    else:
                        engaging_groups.append(engaging_group(content, [same_speaker], date_time, gap, interval, messageID, viewers))
                        break

        if interval == cum_time:

            count += 1
            print "\tRan to interval checking %d" %count

            trackings = []

            if len(cooccur.recorded_words) == 0:

                pass

            else:

                words, docs, checked, number_of_words, number_of_docs = prepare_co_occurrence(cooccur.recorded_words)

                Y = scipy.sparse.csc_matrix((checked, (words, docs)), shape = (number_of_words,  number_of_docs))

                Yc = (Y * Y.T)
                Yc.setdiag(0)
                sqrt_avg_degrees = math.sqrt(np.mean(Yc.sum(axis = 1)))

                eigenvalues = NB_Spectral(Yc)
                print "\tLargest 2: %s" %eigenvalues
                cooccur_eval, group_eval, approx = product_eval1_eval2(eigenvalues, engaging_groups, tree_para, sqrt_avg_degrees)

                print "\tCooccur Eigen: %f, Avg Deg: %f, Engaging Eigen: %f, Approx: %f \n" %(cooccur_eval, sqrt_avg_degrees, group_eval, approx)

                x_axis_time.append(date_time)
                y_axis_engaging.append(approx)
                background.append(cooccur_eval)
                groups_engaging.append(group_eval)
                supervised_pairs.append((happened, approx))

        if cum_time > interval:

            dynamic_del_count += 1

            print "\tEntering dynamic deletion %d \n" %dynamic_del_count

            speeches.remove_old(date_time, interval)
            cooccur.remove_old(date_time, interval)

            for e in engaging_groups:
                e.remove_old(date_time, interval)

            engaging_groups = [e for e in engaging_groups if len(e.recorded_speakers) > 0]

            while sum_time([x for x in recent_history]) > interval:
                recent_history.popleft()

    run_time = (time.time() - supervised_start)/60
    print "Took: %f minutes" %(run_time)
    print "Speed: %f data points per minute" %(the_chunk/run_time)

    output_raw = zip(x_axis_time, y_axis_engaging, background, groups_engaging)
    pd.DataFrame(output_raw).to_excel(raw_result, header = False, index = False)

    # Supervised Detection Starts

    x_axis_recall = []
    y_axis_precision = []
    scales = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

    for t in scales:

        predicted = 0
        predicted_sum = 0
        relevant_retrieved = 0
        relevant = 0

        for sp in supervised_pairs:

            relevant += sp[0]
            fact = sp[0]

            if sp[1] > round(t,4):
                predicted = 1
                predicted_sum += 1

                if (predicted == 1) and (fact == 1):
                    relevant_retrieved += 1
            else:
                predicted = 0

        if relevant > 0:
            x_axis_recall.append(relevant_retrieved/float(relevant))
        else:
            x_axis_recall.append(0)

        if predicted_sum > 0:
            y_axis_precision.append(relevant_retrieved/float(predicted_sum))
        else:
            y_axis_precision.append(0)

    # Supervised Result Output

    with open(precision_recall, 'a') as pr:
        pr.write("Number of data points: %d \n" %(the_chunk))
        pr.write("Took: %f minutes\n" %(run_time))
        pr.write("Speed: %f data points per minute\n" %(the_chunk/run_time))
        pr.write("Recall: %s \n" %x_axis_recall)
        pr.write("Precision: %s \n" %y_axis_precision)

    plt.plot(x_axis_recall, y_axis_precision, '-o')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig(the_graph)

if __name__ == '__main__':
    main()
    print "The program is done.\n"
