from lsh import LSH
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

import pandas as pd
import numpy as np
import math
import time
import sys

pd.options.mode.chained_assignment = None 

PERM = 100 #number of independent hash functions (e.g. 100) for computing signatures' matrix
SCALING_FACTOR = 10000

QUERY_WEIGHT = 0.6
USER_WEIGHT = 0.4
DEFAULT_MEAN = 60

features = ["name","address","age","occupation"]

class Recommender:

	def __init__(self, users, queries, dataset, ratings):
		self.users = users.to_numpy()
		self.queries = queries.to_numpy()
		self.dataset = dataset
		self.queriesIDs = list(ratings.columns)
		self.usersIDs = list(ratings.index.values)
		self.ratings = ratings.to_numpy()
		self.tupleCount = {}

	def compute_shingles(self):

		initial = time.time()

		shingles_matrix = {}

		for q in range(len(self.queries)):

			filteredDataset = self.dataset
			for f in range(len(features)):
				if self.queries[q][f] != "":
					if self.queries[q][f].isdigit():
						filteredDataset = filteredDataset[filteredDataset[features[f]] == int(self.queries[q][f])]
					else:
						filteredDataset = filteredDataset[filteredDataset[features[f]] == self.queries[q][f]]
			
			self.tupleCount[q] = len(filteredDataset.index.values)

			for ind in filteredDataset.index.values:
				if not ind in shingles_matrix:
					shingles_matrix[ind] = set()
				shingles_matrix[ind].add(q)

		print(str(round(time.time() - initial, 3)) + "s for shingles_matrix")
		return shingles_matrix

	def compute_signatures(self):
		
		shingles_matrix = self.compute_shingles()

		initial = time.time()

		sign_mat = np.empty((PERM, len(self.queries)))
		sign_mat[:] = -1

		queryList = set()
		perm = np.arange(len(self.dataset))

		for i in range(PERM):
			np.random.shuffle(perm)
			queryList.clear()

			partition = np.argsort(perm)

			while len(self.queries) != len(queryList) and len(partition) != 0:

				index_min = partition[0]
				if index_min in shingles_matrix:
					
					queryIndex = list(queryList.difference(shingles_matrix[index_min]))
					sign_mat[i][queryIndex] = perm[index_min]
					queryList.update(queryIndex)

				partition = np.delete(partition, 0)

		print(str(round(time.time() - initial, 3)) + "s for signature_matrix")

		return sign_mat.transpose() #get column, i.e. signature

	def compute_querySimilarities(self):

		signatures = self.compute_signatures()

		initial = time.time()
		
		# SPARSE MATRIX APPROACH 43.6s
		sig_sparse = csr_matrix(signatures)
		query_sim = cosine_similarity(sig_sparse)

		#QUERY_THRESH = np.percentile(query_sim, 97)

		#query_sim[query_sim < QUERY_THRESH] = 0
		query_sim = np.array(query_sim * SCALING_FACTOR, dtype='int16')
		np.fill_diagonal(query_sim, 0)

		CANDIDATES = round(math.log(len(self.queries), 1.5))
		top_query = {}

		for i in range(len(query_sim)):	
			if all(s == -1 for s in signatures[i]):
				query_sim[i] = 0
				query_sim[:,i] = 0

			top_query[i] = np.argsort(query_sim[i])[::-1][0:CANDIDATES]

		print(str(round(time.time() - initial, 3)) + "s for queries_similarity")
		return query_sim, top_query

	def compute_userSimilarities(self):

		initial = time.time()

		norm = np.copy(self.ratings)
		norm[np.isnan(norm)] = 0

		user_sim = np.corrcoef(norm)

		#USER_THRESH = np.percentile(user_sim, 97)

		#user_sim[user_sim < USER_THRESH] = 0
		user_sim = np.array(user_sim * SCALING_FACTOR, dtype='int16')
		np.fill_diagonal(user_sim, 0)

		CANDIDATES = round(math.log(len(self.users), 1.5))
		top_users = {}

		for i in range(len(self.users)):
			top_users[i] = np.argsort(user_sim[i])[::-1][0:CANDIDATES]

		print(str(round(time.time() - initial, 3)) + "s for users_similarity")
		
		return user_sim, top_users

	def compute_scores(self):

		querySimilarities, topQueryIndexes = self.compute_querySimilarities()
		userSimilarities, topUserIndexes = self.compute_userSimilarities()

		scores_to_predict = np.array(np.where(np.isnan(self.ratings))).transpose()

		queryPrediction = 0
		userPrediction = 0
		finalPredictions = np.copy(self.ratings)

		initial = time.time()

		count = 0
		for i, j in scores_to_predict:

			# COLLABORATIVE FILTERING QUERY-QUERY
			userRating = np.array(self.ratings[i][topQueryIndexes[j]])
			simScores = np.array(querySimilarities[j][topQueryIndexes[j]])
			queryPrediction = self.nan_average(userRating, simScores)

			# COLLABORATIVE FILTERING USER-USER
			userRating = np.array(self.ratings.transpose()[j][topUserIndexes[i]])
			simScores = np.array(userSimilarities[i][topUserIndexes[i]])
			userPrediction = self.nan_average(userRating, simScores)

			# HYBRID PREDICTIONS
			if userPrediction == 0 and queryPrediction == 0:
				finalPredictions[i][j] = np.nan #cannot find a predictable value
			elif userPrediction == 0:
				finalPredictions[i][j] = round(queryPrediction * (QUERY_WEIGHT + (USER_WEIGHT*0.5)) + DEFAULT_MEAN * (USER_WEIGHT*0.5))
			elif queryPrediction == 0:
				finalPredictions[i][j] = round(userPrediction * (USER_WEIGHT + (QUERY_WEIGHT*0.5)) + DEFAULT_MEAN * (QUERY_WEIGHT*0.5))
			else:
				finalPredictions[i][j] = round(queryPrediction * QUERY_WEIGHT + userPrediction * USER_WEIGHT)

			count += 1

			if count % 10000 == 0:
				print("{} / {} [{}s]".format(count, len(scores_to_predict), round(time.time() - initial, 3)))
			

		print(str(round(time.time() - initial, 3)) + "s for weighted averages")

		finalPredictions = pd.DataFrame(finalPredictions, columns = self.queriesIDs, index = self.usersIDs)
		scores_missed = np.array(np.where(np.asanyarray(np.isnan(finalPredictions)))).transpose()

		return len(scores_to_predict), finalPredictions, len(scores_missed)

	def suggest_queries(self, predictions):
		
		predictions = predictions.to_numpy().transpose()
		suggestions = []

		initial = time.time()
		for q in range(len(predictions)):
			suggestions.append((self.queriesIDs[q], np.nanmean(predictions[q]), self.tupleCount[q]))

		suggestions.sort(key=lambda item: (item[1], item[2]), reverse=True)

		print(str(round(time.time() - initial, 3)) + "s for queries suggestion")

		return suggestions

	def nan_average(self, values, weights):
		weights = weights / SCALING_FACTOR
		weightSum = ((~np.isnan(values)) * weights).sum()
		return 0.0 if weightSum == 0.0 else np.nansum(values * weights) / weightSum




	    
	   
	    