#!/usr/bin/env python3
import requests
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import math

#"test"
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
	#Do the thing

	return query

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


def dcg(score):
	dcg = score[0][1]
	for single_score in score[1:]:
		dcg += single_score[1]/math.log(single_score[0])
	return dcg


def evaluate(query): #,relevance):
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

	print(dcg(score_pos))


def main():
	#Load query and relevance data
	queries = ['Amsterdam','Mclaren','Python','Huygens']
	relevance = []

	#(Optional) Split into folds

	#(Optional) Training

	evaluate("Who is the mayor of Berlin")

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

#If we want to do machine learning, we would need a lot of data. What data would we need to train on?
