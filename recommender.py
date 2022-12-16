# query similarity
from lsh import LSH

# clustering 
from sklearn.cluster import Birch
from sklearn.preprocessing import StandardScaler

# mathematical computations
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr #alternative for pearson similarity: not working on matrix

# dataframe to handle csv
from datatable import dt, f
import pandas as pd

# general imports
import numpy as np
import matplotlib.pyplot as plt
import itertools
import collections
import math
import time
import gc

pd.options.mode.chained_assignment = None 

# constants
PERM = 120 # number of independent hash functions (e.g. 100) for computing signatures' matrix
BAND = 40

QUERY_WEIGHT = 0.6
USER_WEIGHT = 0.4
DEFAULT_MEAN = 60

class Recommender:

	def __init__(self, users, queries, queriesIDs, dataset, ratings):
		self.allUsers = users.to_numpy()
		self.usersIDs = list(itertools.chain.from_iterable(ratings['user'].to_numpy()))

		self.queries = queries.to_numpy()
		self.queriesIDs = queriesIDs
		
		dataset[:] = dt.str32
		self.datasetFeatures = list(dataset.names)[1::]
		self.dataset = dataset.to_pandas()
		self.tupleCount = {}
		
		ratings.replace({None: 0}) #replace NaN with 0
		del ratings[:, ['user']] 
		self.ratings = ratings.to_numpy()

	def compute_shingles(self):

		print("Dataset : {}, Total queries: {}".format(self.dataset.shape[0], len(self.queries)))

		initial = time.time()

		shingles_matrix = {}

		count = 0
		for q in range(len(self.queries)):

			#query = ""
			filteredDataset = self.dataset
			for ft in range(len(self.datasetFeatures)):
				if self.queries[q][ft] != "":
					filteredDataset = filteredDataset[filteredDataset[self.datasetFeatures[ft]] == self.queries[q][ft]]
					#if query == "":
						#query = self.datasetFeatures[ft] + '=="' + self.queries[q][ft] + '"'
					#else:
						#query += ' | ' + self.datasetFeatures[ft] + '=="' + self.queries[q][ft] + '"'

			#filteredDataset = self.dataset.query(query, inplace=False)

			self.tupleCount[q] = len(filteredDataset.index.values)

			for ind in filteredDataset.index.values:
				if not ind in shingles_matrix.keys():
					shingles_matrix[ind] = []
				shingles_matrix[ind].append(q)	

			count += 1

			if count % 100 == 0:
				print("{} / {} [{}s]".format(count, len(self.queries), round(time.time() - initial, 3)))

		shingles_matrix = dict(collections.OrderedDict(sorted(shingles_matrix.items())))

		print(str(round(time.time() - initial, 3)) + "s for shingles_matrix")
		return shingles_matrix

	def compute_signatures(self):
		
		shingles_matrix = self.compute_shingles()

		print("\nPermutations: {}".format(PERM))

		initial = time.time()
		
		drows, dcols = self.dataset.shape

		queryList = set()
		perm = [i for i in range(drows)]

		sign_mat = np.empty((PERM, len(self.queries)))
		sign_mat[:] = -1

		empty_sign = [False] * len(self.queries)

		count = 0

		for i in range(PERM):
			np.random.shuffle(perm)
			queryList.clear()

			partition = np.argsort(perm)

			while len(self.queries) != len(queryList) and len(partition) != 0:
				index_min = partition[0]
				if index_min in shingles_matrix:
					for q in shingles_matrix[index_min]:
						if not q in queryList:
							queryList.add(q)
							sign_mat[i][q] = perm[index_min]
				partition = np.delete(partition, 0)

			count += 1

			if count % 10 == 0:
				print("{} / {} [{}s]".format(count, PERM, round(time.time() - initial, 3)))

		sign_mat = sign_mat.transpose() #get column, i.e. signature

		for s in range(len(sign_mat)):
			if np.all(sign_mat[s] == -1):
				empty_sign[s] = True

		del shingles_matrix
		gc.collect()

		print(str(round(time.time() - initial, 3)) + "s for signature_matrix")

		return sign_mat, empty_sign

	def compute_querySimilarities(self):

		signatures, empty_signatures = self.compute_signatures()
		
		MAX_CANDIDATES = round(math.log(len(self.queries), 1.5))

		print("\nMax query candidates: {}, Total queries: {}".format(MAX_CANDIDATES, len(self.queries)))

		initial = time.time()
		
		lsh = LSH(BAND)

		for sig in signatures:
			lsh.compute_buckets(sig)

		candidates = lsh.get_candidates(signatures)

		print("Candidates pair: {}".format(len(candidates)))

		query_sim = {}
		available_query = [False] * len(self.queries)

		for i, j in candidates:
			if not (empty_signatures[i] or empty_signatures[j]):
				sim = round(max(0, 1 - distance.cosine(signatures[i], signatures[j])), 3)

				if not i in query_sim:
					query_sim[i] = {}
					query_sim[i]['indexes'] = []
					query_sim[i]['values'] = []
				
				if not j in query_sim:
					query_sim[j] = {}
					query_sim[j]['indexes'] = []
					query_sim[j]['values'] = []
				
				query_sim[i]['indexes'].append(j)
				query_sim[i]['values'].append(sim)

				query_sim[j]['indexes'].append(i)
				query_sim[j]['values'].append(sim)

				available_query[i] = True
				available_query[j] = True

		query_sim = dict(collections.OrderedDict(sorted(query_sim.items())))

		for i in query_sim.keys():
			ind = np.argsort(query_sim[i]["values"])[::-1][0:MAX_CANDIDATES]

			query_sim[i]["indexes"] = np.array(query_sim[i]["indexes"])[ind]
			query_sim[i]["values"] = np.array(query_sim[i]["values"])[ind]

		print(str(round(time.time() - initial, 3)) + "s for queries_similarity")

		del signatures, empty_signatures
		gc.collect()

		print(query_sim)

		return query_sim, available_query

	def compute_userSimilarities(self):

		MAX_CANDIDATES = round(math.log(len(self.usersIDs), 1.5))
		cluster_count = round(len(self.usersIDs) * 0.30)

		initial = time.time()
		
		scaler = StandardScaler()
		normScores = scaler.fit_transform(self.ratings)

		model = Birch(n_clusters = cluster_count)
		model.fit(normScores)
		clusters = model.predict(normScores)

		print("\nCluster count: {}, Total users: {}".format(cluster_count, len(self.usersIDs)))

		print(str(round(time.time() - initial, 3)) + "s for data preparation")

		initial = time.time()

		user_sim = {}

		for i in clusters:
			if not i in user_sim:
				user_sim[i] = {}
				c_scores = np.array(self.ratings[clusters == i])

				for s in range(len(c_scores)):
					mean = np.mean(c_scores[s][c_scores[s] != 0])
					c_scores[s][c_scores[s] != 0] = c_scores[s][c_scores[s] != 0] - mean

				user_sim[i]["indexes"] = np.where(clusters == i)[0]
				user_sim[i]["values"] = np.around(cosine_similarity(c_scores), 3)
				np.fill_diagonal(user_sim[i]["values"], 0)
				user_sim[i]["values"][user_sim[i]["values"] < 0] = 0

		top_users = {}

		for i in range(len(self.usersIDs)):
			clusterID = clusters[i]
			userID = np.where(user_sim[clusterID]["indexes"] == i)
			userID = userID[0][0] if len(userID) == 1 else -1
			if userID != -1:
				top_users[i] = np.argsort(user_sim[clusterID]["values"][userID])[::-1][0:MAX_CANDIDATES]
			else:
				top_users[i] = []

		print(str(round(time.time() - initial, 3)) + "s for users_similarity")

		return user_sim, top_users, clusters

	def compute_scores(self):

		#lot of memory usage because of precomputation of all similarities
		#querySimilarities, topQueryIndexes = self.compute_querySimilarities()
		querySimilarities, available_query = self.compute_querySimilarities()
		userSimilarities, topUserIndexes, clusters = self.compute_userSimilarities()

		scores_to_predict = np.array(np.where(self.ratings == 0)).transpose()

		print(scores_to_predict)

		queryPrediction = 0
		userPrediction = 0

		initial = time.time()

		count = 0

		for i, j in scores_to_predict:

			# COLLABORATIVE FILTERING QUERY-QUERY 
			if available_query[j]:
				userRating = np.array(self.ratings[i][querySimilarities[j]["indexes"]])
				simScores = np.array(querySimilarities[j]["values"])
				simScores[userRating == 0] = 0
				weightSum = np.sum(simScores)

				if weightSum == 0:
					queryPrediction = 0
				else:
					queryPrediction = np.sum(userRating * simScores) / weightSum
			else:
				queryPrediction = 0

			# COLLABORATIVE FILTERING USER-USER
			if len(topUserIndexes[i]) > 0:
				clusterID = clusters[i]
				userID = list(userSimilarities[clusterID]["indexes"]).index(i)
				userRating = np.array(self.ratings.transpose()[j][topUserIndexes[i]])
				simScores = np.array(userSimilarities[clusterID]["values"][userID][topUserIndexes[i]])
				simScores[userRating == 0] = 0
				weightSum = np.sum(simScores)

				if weightSum == 0:
					userPrediction = 0
				else:
					userPrediction = np.sum(userRating * simScores) / weightSum
			else:
				userPrediction = 0

			# HYBRID PREDICTIONS
			#print(userPrediction, queryPrediction)
			if userPrediction == 0 and queryPrediction == 0:
				self.ratings[i][j] = 0 #cannot find a predictable value
			elif userPrediction == 0:
				self.ratings[i][j] = round(queryPrediction * (QUERY_WEIGHT + (USER_WEIGHT*0.5)) + DEFAULT_MEAN * (USER_WEIGHT*0.5))
			elif queryPrediction == 0:
				self.ratings[i][j] = round(userPrediction * (USER_WEIGHT + (QUERY_WEIGHT*0.5)) + DEFAULT_MEAN * (QUERY_WEIGHT*0.5))
			else:
				self.ratings[i][j] = round(queryPrediction * QUERY_WEIGHT + userPrediction * USER_WEIGHT)

			count += 1

			if count % 10000 == 0:
				print("{} / {} [{}s]".format(count, len(scores_to_predict), round(time.time() - initial, 3)))
			

		print(str(round(time.time() - initial, 3)) + "s for weighted averages")

		finalPredictions = pd.DataFrame(self.ratings, columns = self.queriesIDs, index = self.usersIDs)
		scores_missed = np.array(np.where(finalPredictions == 0)).transpose()

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




	    
	   
	    