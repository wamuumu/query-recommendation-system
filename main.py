from recommender import Recommender
from resources import generator
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd
import math, time
import numpy as np

pd.options.mode.chained_assignment = None 

QUERY_WEIGHT = 0.6
USER_WEIGHT = 0.4
DEFAULT_MEAN = 60


def nan_average(A,weights,axis):
	den = ((~np.isnan(A))*weights).sum(axis=axis)
	return 0 if den == 0 else np.nansum(A*weights,axis=axis)/den

if __name__ == "__main__":

	initial = time.time()
	users = pd.read_csv("./resources/output/users.csv", names = ["id"], header = None, engine="pyarrow")
	print(str(time.time() - initial) + "s to read users")

	initial = time.time()
	queries = generator.parse_queries("./resources/output/queries.csv")
	print(str(time.time() - initial) + "s to read queries")

	initial = time.time()
	dataset = pd.read_csv("./resources/output/dataset.csv", names = ["id","name","address","age","occupation"], header = 0, engine="pyarrow")
	print(str(time.time() - initial) + "s to read dataset")

	initial = time.time()
	ratings = pd.read_csv("./resources/output/utility_matrix.csv", engine="pyarrow")
	print(str(time.time() - initial) + "s to read utility matrix")

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

	scores_to_predict = np.array(np.where(np.asanyarray(np.isnan(ratings)))).transpose()

	# QUERY SIMILARITIES USING LSH AND MIN-HASHING
	
	initial = time.time()
	queryPrediction = 0
	userPrediction = 0
	
	finalPredictions = ratings.copy().to_numpy()
	
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

	# 33s with no averages
	count = 0
	for i, j in scores_to_predict:

		# QUERY SIMILARITIES USING LSH AND MIN-HASHING
		if empty_query_weights[j]:
			queryPrediction = 0
		else:
			if qpred[i] == -1:
				queryPrediction = round(nan_average(ratings.iloc[i].to_numpy(), querySimilarities[j], 0))
				qpred[i] = queryPrediction
			else:
				queryPrediction = qpred[i]

		#input("Press Enter to continue...")
		
		# COLLABORATIVE FILTERING USER-USER
		if empty_user_weights[i]:
			userPrediction = 0
		else:
			if upred[j] == -1:
				userPrediction = round(nan_average(ratings.to_numpy().transpose()[j], userSimilarities[i], 0))
				upred[j] = userPrediction
			else:
				userPrediction = upred[j]


		# HYBRID PREDICTIONS
		if userPrediction == 0 and queryPrediction == 0:
			finalPredictions[i][j] = np.nan #cannot find a predictable value
		elif userPrediction == 0:
			finalPredictions[i][j] =  round(queryPrediction * (QUERY_WEIGHT+  (USER_WEIGHT*0.5)) + DEFAULT_MEAN * (USER_WEIGHT*0.5))
		elif queryPrediction == 0:
			finalPredictions[i][j] =  round(userPrediction * (USER_WEIGHT + (QUERY_WEIGHT*0.5)) + DEFAULT_MEAN * (QUERY_WEIGHT*0.5))
		else:
			finalPredictions[i][j] = round(queryPrediction * QUERY_WEIGHT + userPrediction * USER_WEIGHT)

		
		count += 1

		print(str((count / len(scores_to_predict))*100) + "%")

	print(str(time.time() - initial) + "s to calculate weighted averages")

	
	finalPredictions = pd.DataFrame(finalPredictions, columns = queriesIDs, index = usersIDs)
	score_missed = np.array(np.where(np.asanyarray(np.isnan(finalPredictions)))).transpose()

	print("\nINITIAL RATINGS [{} scores to predict]:".format(len(scores_to_predict)))	
	print(ratings)

	print("\nFINAL PREDICTIONS [{} scores missed - {}%]:".format(len(score_missed), round(len(score_missed) / len(scores_to_predict), 2) * 100))	
	print(finalPredictions)

	'''
	print(time.time() - initial)
	csv_rows = finalPredictions.to_numpy().tolist()
	
	for ind in range(len(list(finalPredictions.index.values))):
		csv_rows[ind].insert(0, finalPredictions.index.values[ind])

	print(len(usersIDs), len(queriesIDs), len(usersIDs) * len(queriesIDs))
	generator.write_csv("output", finalPredictions.columns, csv_rows)
	'''

	exit(0)