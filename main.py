import pandas as pd
from recommender import Recommender
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import math
import numpy as np
from resources import generator

pd.options.mode.chained_assignment = None 

QUERY_THRESH = 0.4
USER_THRESH = 0.4
QUERY_WEIGHT = 0.6
USER_WEIGHT = 0.4
DEFAULT_MEAN = 60


def parse_queries(path:str):

	data = {}

	with open(path) as f:
		for row in f:
			row = row.rstrip('\n')
			values = row.split(",")
			index = values[0]
			values = values[1::]

			data[index] = []
			for attr in values:
				val = attr.split("=")
				if val[0] != "age" and val[0] != "id":
					val[1] = '"' + val[1] + '"'
				val = val[0] + "=" + val[1]
				data[index].append(val)

	return data

def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        elif isinstance(x, int):
        	return str(x)
        else:
            return ''

def brute_similarity(recommender, queries, query1ID, query2ID, mat):
	
	d1 = recommender.compute_query(queries[query1ID])
	d2 = recommender.compute_query(queries[query2ID])

	union_dfs = pd.concat([d1, d2]).drop_duplicates(subset=['id'])

	#total = max([len(d1.index), len(d2.index)])
	total = len(union_dfs.index)

	'''print(d1)
	print()
	print(d2)
	print()'''

	thresh = 0.5
	count = 0

	for id1 in d1['id']:
		for id2 in d2['id']:
			#print("comparing " + str(id1) + " and " + str(id2))
			if mat[id1-1][id2-1] >= thresh:
				count += 1
				break

	return round(count / total, 2)

def create_text(x):
	return x['name'] + ' ' + x['age'] + ' ' + x['address'] + ' ' + x['occupation']

if __name__ == "__main__":
	
	features = ["name","address","age","occupation"]

	users = pd.read_csv("./resources/output/users.csv", names = ["id"], header = None)
	queries = parse_queries("./resources/output/queries.csv")
	dataset = pd.read_csv("./resources/output/dataset.csv", names = ["id","name","address","age","occupation"], header = 0)
	ratings = pd.read_csv("./resources/output/utility_matrix.csv")

	queriesIDs = list(ratings.columns)
	usersIDs = list(ratings.index.values)

	'''print(users)
	print(dataset)
	print(ratings)
	
	print(queriesIDs)
	print(usersIDs)'''

	recommender = Recommender(users, queries, dataset, ratings)

	querySimilarities = recommender.compute_querySimilarities()
	userSimilarities = recommender.compute_userSimilarities()

	print(querySimilarities)
	print()
	print("\nINITIAL RATINGS:")	
	print(ratings)

	nanIdexes = []
	for u in range(len(usersIDs)):
		for key, value in ratings.iteritems():
			if math.isnan(value[usersIDs[u]]):
				#print(u, queriesIDs.index(key))
				nanIdexes.append((u, queriesIDs.index(key)))

	
	# QUERY SIMILARITIES USING LSH AND MIN-HASHING

	queryPredictions = ratings.copy()
	for i, j in nanIdexes:
		weightSum = 0
		weightedAverage = 0
		for s in range(len(querySimilarities[j])):
			if querySimilarities[j][s] >= QUERY_THRESH and s != j and not math.isnan(ratings[queriesIDs[s]][i]):
				weightedAverage += querySimilarities[j][s] * ratings[queriesIDs[s]][i]
				weightSum += querySimilarities[j][s]
		if weightSum != 0 and weightedAverage != 0:
			queryPredictions[queriesIDs[j]][i] = round(weightedAverage / weightSum)

	print("\nQUERY PREDICTIONS")	
	print(queryPredictions)

	# COLLABORATIVE-FILTERING USER-USER

	userPredictions = ratings.copy()
	for i, j in nanIdexes:
		weightSum = 0
		weightedAverage = 0
		for u in range(len(usersIDs)):
			if userSimilarities[i][u] >= USER_THRESH and u != i and not math.isnan(ratings[queriesIDs[j]][u]):
				weightedAverage += ratings[queriesIDs[j]][u] * userSimilarities[i][u]
				weightSum += userSimilarities[i][u]
		if weightSum != 0 and weightedAverage != 0:
			userPredictions[queriesIDs[j]][i] = round(weightedAverage / weightSum)

	print("\nUSER PREDICTIONS")	
	print(userPredictions)

	# HYBRID PREDICTIONS

	finalPredictions = ratings.copy()
	for i, j in nanIdexes:
		if math.isnan(userPredictions[queriesIDs[j]][i]) and math.isnan(queryPredictions[queriesIDs[j]][i]):
			finalPredictions[queriesIDs[j]][i] = -1
		elif math.isnan(userPredictions[queriesIDs[j]][i]):
			finalPredictions[queriesIDs[j]][i] =  round(queryPredictions[queriesIDs[j]][i] * (QUERY_WEIGHT + (USER_WEIGHT*0.5)) + DEFAULT_MEAN * (USER_WEIGHT*0.5))
		elif math.isnan(queryPredictions[queriesIDs[j]][i]):
			finalPredictions[queriesIDs[j]][i] =  round(userPredictions[queriesIDs[j]][i] * (USER_WEIGHT + (QUERY_WEIGHT*0.5)) + DEFAULT_MEAN * (QUERY_WEIGHT*0.5))
		else:
			finalPredictions[queriesIDs[j]][i] = round(queryPredictions[queriesIDs[j]][i] * QUERY_WEIGHT + userPredictions[queriesIDs[j]][i] * USER_WEIGHT)
	
	print("\nFINAL PREDICTIONS")	
	print(finalPredictions)
	print()	

	csv_rows = finalPredictions.to_numpy().tolist()
	
	for ind in range(len(list(finalPredictions.index.values))):
		csv_rows[ind].insert(0, finalPredictions.index.values[ind])

	print(len(usersIDs), len(queriesIDs), len(usersIDs) * len(queriesIDs))
	generator.write_csv("output", finalPredictions.columns, csv_rows)

	'''print()
	recommender.compute_query(queries['Q1'])
	print()	
	recommender.compute_query(queries['Q2'])
	print()
	recommender.compute_query(queries['Q3'])

	features_dataset = dataset.drop(columns=['id'])
	for feature in features:
		features_dataset[feature] = features_dataset[feature].apply(clean_data)

	features_dataset['desc'] = features_dataset.apply(create_text, axis=1)

	print(features_dataset)

	# In CountVectorizer we only count the number of times a word appears in the document which results in biasing in favour of most frequent words.
	# In TfidfVectorizer we consider overall document weightage of a word. TfidfVectorizer weights the word counts by a measure of how often they appear in the documents.
	count = CountVectorizer(stop_words='english')
	count_matrix = count.fit_transform(features_dataset['desc'])

	# cosine similarity for different document size
	# jaccard for short documents of equal size
	#print(count_matrix)
	cosine_sim = cosine_similarity(count_matrix, count_matrix)

	for r in range(cosine_sim.shape[0]):
		for c in range(cosine_sim.shape[0]):
			cosine_sim[r][c] = round(cosine_sim[r][c], 2)

	print()
	print(cosine_sim)
	print()

	brute = brute_similarity(recommender, queries, 'Q1', 'Q3', cosine_sim)

	print("Brute similarity -> " + str(brute))'''

	exit(0)