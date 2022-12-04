import pandas as pd
from recommender import Recommender
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import math, time
import numpy as np
from resources import generator

pd.options.mode.chained_assignment = None 

QUERY_WEIGHT = 0.6
USER_WEIGHT = 0.4
DEFAULT_MEAN = 60


def nanaverage(A,weights,axis):
    return np.nansum(A*weights,axis=axis)/((~np.isnan(A))*weights).sum(axis=axis)

if __name__ == "__main__":

	users = pd.read_csv("./resources/output/users.csv", names = ["id"], header = None)
	queries = generator.parse_queries("./resources/output/queries.csv")

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

	print("\nINITIAL RATINGS:")	
	print(ratings)

	nanIdexes = np.array(np.where(np.asanyarray(np.isnan(ratings)))).transpose()

	print(nanIdexes)
	print("\n" + str(len(nanIdexes)) + " scores to predict")

	# QUERY SIMILARITIES USING LSH AND MIN-HASHING

	initial = time.time()
	queryPrediction = 0
	userPrediction = 0
	
	supportDataframe = ratings.copy().to_numpy().transpose()
	finalPredictions = ratings.copy().to_numpy()
	ratings = ratings.fillna(0)
	
	qpred = [-1] * len(usersIDs) # 1 value for each user (between queries)
	upred = [-1] * len(queriesIDs) # 1 value for each query (between users)
	empty_query_weights = [False] * len(queriesIDs)
	empty_user_weights = [False] * len(usersIDs)

	for q in range(len(queriesIDs)):
		if not np.any(querySimilarities[q]):
			empty_query_weights[q] = True

	for u in range(len(usersIDs)):
		if not np.any(userSimilarities[u]):
			empty_user_weights[u] = True

	count = 0
	for i, j in nanIdexes:

		# QUERY SIMILARITIES USING LSH AND MIN-HASHING
		if empty_query_weights[j]:
			queryPrediction = 0
		else:
			if qpred[i] == -1:
				queryPrediction = round(np.average(a = ratings.iloc[i].to_numpy(), weights = querySimilarities[j]))
				qpred[i] = queryPrediction
			else:
				queryPrediction = qpred[i]
		
		# COLLABORATIVE FILTERING USER-USER
		if empty_user_weights[i]:
			userPrediction = 0
		else:
			if upred[j] == -1:
				userPrediction = round(nanaverage(supportDataframe[j], userSimilarities[i], 0))
				upred[j] = userPrediction
			else:
				userPrediction = upred[j]


		# HYBRID PREDICTIONS
		if userPrediction == 0 and queryPrediction == 0:
			finalPredictions[i][j] = -1 #cannot find a predictable value
		elif userPrediction == 0:
			finalPredictions[i][j] =  round(queryPrediction * (QUERY_WEIGHT+  (USER_WEIGHT*0.5)) + DEFAULT_MEAN * (USER_WEIGHT*0.5))
		elif queryPrediction == 0:
			finalPredictions[i][j] =  round(userPrediction * (USER_WEIGHT + (QUERY_WEIGHT*0.5)) + DEFAULT_MEAN * (QUERY_WEIGHT*0.5))
		else:
			finalPredictions[i][j] = round(queryPrediction * QUERY_WEIGHT + userPrediction * USER_WEIGHT)

		
		count += 1

		print(str((count / len(nanIdexes))*100) + "%")

	print(time.time() - initial)

	finalPredictions = pd.DataFrame(finalPredictions, columns = queriesIDs, index = usersIDs).astype(int)
	
	print("\nFINAL PREDICTIONS")	
	print(finalPredictions)
	print()	

	'''
	print(time.time() - initial)
	csv_rows = finalPredictions.to_numpy().tolist()
	
	for ind in range(len(list(finalPredictions.index.values))):
		csv_rows[ind].insert(0, finalPredictions.index.values[ind])

	print(len(usersIDs), len(queriesIDs), len(usersIDs) * len(queriesIDs))
	generator.write_csv("output", finalPredictions.columns, csv_rows)
	'''

	exit(0)