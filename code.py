#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from operator import itemgetter
import nltk
import time
import json
import spacy
from spacy import displacy
from collections import Counter
# import en_core_web_sm


from nltk.corpus import wordnet
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import wordpunct_tokenize, sent_tokenize, word_tokenize
from nltk.corpus import stopwords, brown

#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('brown')
#nltk.download('averaged_perceptron_tagger')

PLOTPATH = 'plots'
DATAPATH = 'data'
freqs = nltk.FreqDist(w.lower() for w in brown.words())

import os
if not os.path.exists(PLOTPATH):
    os.makedirs(PLOTPATH)

drive = ''

def avg(data):
    return sum(data)/len(data)


def plot_results(plain,our,measure_id = 2):
    plt.close()

    width = 0.2
    x = np.array(range(len(plain)))
    plt.bar(x-width/2.0,plain[:,measure_id], width = width, color = 'blue', label = 'Plain Query')
    plt.bar(x+width/2.0,our[:,measure_id], width = width, color = 'red' , label = 'Transformed Query')

    plt.xticks(range(len(plain)))
    plt.xlabel('query_id')
    measure_name = ["Precision","Recall","F-measure"][measure_id]
    plt.ylabel(measure_name)

    plt.legend()
    plt.title(measure_name)
    plt.savefig(PLOTPATH+"/"+measure_name+".png")

def plot_averages(plain,our):
    plt.close()

    plain_avg = list(map(np.average,[plain[:,0],plain[:,1],plain[:,2]]))
    our_avg   = list(map(np.average,[our[:,0]  ,our[:,1]  ,our[:,2]]))

    width = 0.4
    x = np.array(range(3))
    plt.bar(x-width/2.0,plain_avg, width = width, color = 'blue', label = 'Plain Query')
    plt.bar(x+width/2.0,our_avg, width = width, color = 'red' , label = 'Transformed Query')

    plt.xticks(range(3))
    plt.xlabel('query_id')
    measure_name = "Averaged Results"
    plt.ylabel(measure_name)

    plt.legend()
    plt.title(measure_name +  " 0 Recall, 1 Precision, 2 F-measure")
    plt.savefig(PLOTPATH+"/"+measure_name+".png")


def plot_errors(scores_data):
    plt.close()
    plt.plot(range(len(scores_data)),[float(mscore)-float(oscore) for _,_,oscore,mscore,_,_ in scores_data])
    plt.title("score distribution")
    plt.savefig(PLOTPATH+"/score_distribution.png")


def removeQuestionWords(word):
    irrelevant_words = ["Give", "List", "What", "Who", "Where", "Which", "How", "me", "all", "In"]

    if (word not in irrelevant_words):
        return word
    else:
        return ""


def entity_retrieval(query):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(query)
    relevant_ents = ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART"]  #, "LANGUAGE"]
    selected_ents = [x.text for x in doc.ents if x.label_ in relevant_ents]
    return selected_ents


def transform(query, stemming=False, stopping=False, qwords=False, synonyms=False, ent_rec=False):

    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(query)
    if stopping:
        tokens = [w for w in word_tokens if not w in stop_words]
    else:
        tokens = word_tokens
    tagged = nltk.pos_tag(tokens)
    if stemming:
        stemmer = SnowballStemmer("english")
        tokens = [stemmer.stem(w) for w in tokens]

    modified_query = ""
    for word in tokens:
        if qwords:
            modified_query += " " + removeQuestionWords(word)
        else:
            modified_query += " " + word

    # modified_query = "".join([char for char in modified_query if char not in ['.',',','?','!']])
    # no difference
    word_tokens = word_tokenize(modified_query)

    ent_additions = ""
    if ent_rec:
        ents = entity_retrieval(query)
        for ent in ents:
            ent_additions += " " + ent

    if synonyms:
        for word in tagged:
            if(word[1] == 'NN' or word[1] == 'NNP' or word[1] == 'NNS'):
                syns = wordnet.synsets(word[0])
                if(syns):
                    syns = list(set([(freqs[syn.lemmas()[0].name().lower()] ,syn.lemmas()[0].name()) for syn in syns]))
                    syns.append((freqs[word[0].lower()] ,word[0]) )
                    syns = sorted(syns, key = lambda x : -x[0]) #-freqs[x[1]])
                    #print(syns)
                    modified_query +=  " " + syns[0][1]
                    #print(syns,word)

    modified_query += ent_additions  # Added these afterwards to avoid synonyms of repeated entities.

    # print(modified_query)
    return modified_query


def get_relevance(query):
    #Compare nordlys with relevance
    with open (drive + "queries-NLP.txt", 'r') as queries:
        for row in queries:
            if query in row:
                code = row.split("\t")[0]

    DB_rels = []
    with open (drive + "qrels-v2.txt", 'r', encoding='utf-8') as qrels:
        for row in qrels:
            if code == row.split("\t")[0]:
                DB_rels.append(row)

    return DB_rels


def calc_ndcg(score):
    if len(score) == 0:
        ndcg = 0
    else:
        dcg = score[0][1]
        sorted_score = sorted(score, key=itemgetter(1), reverse=True)
        # print(sorted_score)
        idcg = sorted_score[0][1]
        for i in range(1, len(score)):
            #print(score[i][0])
            dcg += score[i][1]/(math.log2(score[i][0]) + 1)
            idcg += sorted_score[i][1]/(math.log2(score[i][0]) + 1)
        ndcg = dcg/idcg if idcg > 0 else 0
        #print(dcg)
    return ndcg


def evaluate(query,original_query = None): #,relevance):
    if not original_query:
        original_query = query

    #Query nordlys
    #REST API documentation https://nordlys.readthedocs.io/en/latest/restful_api.html
    base_url = 'http://api.nordlys.cc/'
    parameters = 'er?1st_num_docs=20&model=lm&q='

    response = requests.get(base_url + parameters + query)
    if response.status_code != 200:
        raise ApiError('GET /er?{}/ {}'.format(parameters + query, response.status_code))

    #Access individual result
    #Example json data: http://api.nordlys.cc/er?1st_num_docs=20&model=lm&q=Amsterdam
    results = response.json()['results']
    entities = []
    for i in results:
        result = results[i]
        entities.append(result['entity'])
        #print('{} {}'.format(result['entity'], result['score']))

    score_pos = []
    DB_rels = get_relevance(original_query)
    #print(DB_rels)
    for i in range(len(entities)):
        for rel in DB_rels:
            if entities[i] in rel:
                #print(rel)
                score = int(rel[-2:-1])
                score_pos.append((i+1, score))
    # print(score_pos)

    ndcg = calc_ndcg(score_pos)
    #print("Query: " + query + " \tNDCG = {:f}".format(ndcg))
    return query, ndcg


def main():
    start_time = time.time()

    #Load query and relevance data
    with open (drive + "queries-NLP.txt", 'r') as query_files:
        queries = [q.split("\t")[1][:-1] for q in query_files]


    # queries = ["Who composed the music for Harold and Maude?"]
    #
    modified_queries, oscores, mscores, avgoscores = [],[],[],[]
    avgstemscores, avgstopscores, avgqscores, avgsynscores, avgentscores= [],[],[],[],[]
    stemscores, stopscores, qscores, synscores, entscores, combscores = [],[],[],[],[],[]

    for query in queries:
        stem = transform(query, stemming=True)  # Stemming
        stop = transform(query, stopping=True)  # Stopping
        qwords = transform(query, qwords=True)  # Question Word Removal
        syn = transform(query, synonyms=True)   # Added synonyms
        ent = transform(query, ent_rec=True)    # Added entities
        # comb = transform(query, stemming=False, stopping=True, qwords=True, synonyms=True) #, ent_rec=True)

        # modified_queries.append(modified)

        _,oscore = evaluate(query)
        _,stemscore = evaluate(stem, query)
        _,stopscore = evaluate(stop, query)
        _,qscore = evaluate(qwords, query)
        _,synscore = evaluate(syn, query)
        _,entscore = evaluate(ent, query)
        # _,combscore = evaluate(comb, query)

        oscores.append(oscore)
        stemscores.append(stemscore)
        stopscores.append(stopscore)
        qscores.append(qscore)
        synscores.append(synscore)
        entscores.append(entscore)
        # combscores.append(combscore)

        avgoscores.append(avg(oscores))
        avgstemscores.append(avg(stemscores))
        avgstopscores.append(avg(stopscores))
        avgqscores.append(avg(qscores))
        avgsynscores.append(avg(synscores))
        avgentscores.append(avg(entscores))
        # if (mscore - oscore < -0.05): #print only queries which decrease the score
        #     print("average increase: " + str(avg(mscores) - avg(oscores))[:6] + '\t'+ query  )
        #     print("modified query:","\t\t", modified)
        #     print(str(oscore)[:6],str(synscores)[:6])
        #     print("")

        print("Averages: Original: {:4f}".format(avgoscores[-1]) + "\t stem: {:4f}".format(avgstemscores[-1]) +
              "\t stop: {:4f}".format(avgstopscores[-1]) + "\t q: {:4f}".format(avgqscores[-1]) +
              "\t syn: {:4f}".format(avgsynscores[-1]), "\t ent: {:4f}".format(avgentscores[-1]))

    print("---- Evaluation and transformation time: {:3f} seconds ----".format(time.time() - start_time))

    # scores_data = list(zip(queries,modified_queries,oscores,mscores,avgoscores,avgmscores))
    # scores_data_np = np.asarray(sorted(scores_data,key = lambda x :  -(float(x[3]) - float(x[2])))) #sort on increase

    oscores_sorted = np.asarray(sorted(oscores))  #, reverse=True))
    stemscores_sorted = np.asarray(sorted(stemscores))  #, reverse=True))
    stopscores_sorted = np.asarray(sorted(stopscores))  #, reverse=True))
    qscores_sorted = np.asarray(sorted(qscores))  #, reverse=True))
    synscores_sorted = np.asarray(sorted(synscores))  #, reverse=True))
    entscores_sorted = np.asarray(sorted(entscores))  #, reverse=True))

    x = np.arange(len(oscores))

    plt.plot(x, oscores_sorted, label="Original")
    plt.plot(x, stemscores_sorted, label="Stemming")
    plt.plot(x, stopscores_sorted, label="Stopping")
    plt.plot(x, qscores_sorted, label="Question word removal")
    plt.plot(x, synscores_sorted, label="Synonyms")
    plt.plot(x, entscores_sorted, label="Entity Recognition")
    plt.legend()
    plt.savefig(PLOTPATH + "/methods.png")
    plt.show()

    avgs = np.array([avg(oscores), avg(stemscores), avg(stopscores), avg(qscores), avg(synscores), avg(entscores)])

    # np.save(DATAPATH + "/avgs_methods", avgs)

    # with open('scores_data.json', 'w') as outfile:
    #     json.dump(scores_data, outfile)
    # plot_errors(scores_data_np)


if __name__ == "__main__":
    main()

# Testing entity recognition
# with open (drive + "queries-NLP.txt", 'r') as query_files:
#     queries = [q.split("\t")[1][:-1] for q in query_files]
#     for i in range(len(queries)):
#         print(queries[i])
#         print(entity_retrieval(queries[i]))







