#!/usr/bin/env python3
import requests
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from operator import itemgetter
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import wordpunct_tokenize, sent_tokenize, word_tokenize
from nltk.corpus import stopwords


PLOTPATH = 'plots'

import os
if not os.path.exists(PLOTPATH):
    os.makedirs(PLOTPATH)



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

	print(plain)
	print(plain_avg)
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


def transform(query):
    nltk.download('stopwords')
    nltk.download('punkt')
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(query)
    tokens = [w for w in word_tokens if not w in stop_words]
    stemmer = SnowballStemmer("english")

    modified_query = ""
    for word in tokens:
        modified_query += " " + stemmer.stem(word)

    print("MODIFIED QUERY ", modified_query)
    return modified_query

def get_relevance(query):
	#Compare nordlys with relevance
	with open ("queries-v2.txt", 'r') as queries:
		for row in queries:
			if query in row:
				code = row.split("\t")[0]

	DB_rels = []
	with open ("qrels-v2.txt", 'r') as qrels:
		for row in qrels:
			if code in row:
				DB_rels.append(row)

	return DB_rels


def calc_ndcg(score):
	dcg = score[0][1]
	sorted_score = sorted(score, key=itemgetter(1), reverse=True)
	# print(sorted_score)
	idcg = sorted_score[0][1]
	for i in range(1, len(score)):
		dcg += score[i][1]/math.log(score[i][0])
		idcg += sorted_score[i][1]/math.log(score[i][0])
	ndcg = dcg/idcg
	print(dcg)
	return ndcg


def evaluate(query,original_query = None): #,relevance):
    if(not original_query)
        original_query = query
        
	#Query nordlys
	#REST API documentation https://nordlys.readthedocs.io/en/latest/restful_api.html
	base_url = 'http://api.nordlys.cc/'
	parameters = 'er?1st_num_docs=30&model=lm&q='

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
		# print('{} {}'.format(result['entity'], result['score']))

	score_pos = []
	DB_rels = get_relevance(query)
	for i in range(len(entities)):
		for rel in DB_rels:
			if entities[i] in rel:
				score = int(rel[-2:-1])
				score_pos.append((i+1, score))
				print(score_pos[-1])

	ndcg = calc_ndcg(score_pos)
	print("Query: " + query + " \tNDCG = {:f}".format(ndcg))
	return query, ndcg


def main():
    #Load query and relevance data
    queries = ['Amsterdam','Mclaren','Python','Huygens']
    relevance = []



    #(Optional) Split into folds

    #(Optional) Training

    evaluate("Who is the mayor of Berlin","Who is the mayor of Berlin")

    #Run
    # plain_results = []
    # our_results = []
    # for query in queries:
    # 	print(query)
    # 	plain_results.append(evaluate(query ,relevance))
    # 	our_results.append(evaluate(transform(query) ,relevance))
    #
    # #plots
    # plain_results, our_results = np.array(plain_results) ,np.array(our_results)
    # plot_results(plain_results, our_results,0)
    # plot_results(plain_results, our_results,1)
    # plot_results(plain_results, our_results,2)
    # plot_averages(plain_results, our_results)

if __name__ == "__main__":
    main()
    transform("hi how cats cat catty you doin")
