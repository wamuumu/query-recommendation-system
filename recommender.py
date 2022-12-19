# query similarity
from lsh import LSH

# clustering 
from sklearn.cluster import Birch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# mathematical computations
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr #alternative for pearson similarity: not working on matrix

# dataframes to handle csv
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
BAND = 40 # number of bands in a signature

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

		drows, dcols =self.dataset.shape

		print("Dataset : {}, Total queries: {}".format(drows, len(self.queries)))

		initial = time.time()

		shingles_matrix = {}

		for d in range(drows):
			shingles_matrix[d] = []

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
				shingles_matrix[ind].append(q)

			count += 1

			if count % round(len(self.queries) * 0.1) == 0:
				print("{} / {} [{}s]".format(count, len(self.queries), round(time.time() - initial, 3)))

		#shingles_matrix = dict(collections.OrderedDict(sorted(shingles_matrix.items())))

		print(str(round(time.time() - initial, 3)) + "s for shingles_matrix")
		return shingles_matrix

	def compute_signatures(self):
		
		shingles_matrix = self.compute_shingles()

		print("\nPermutations: {}".format(PERM))

		initial = time.time()
		
		drows, dcols = self.dataset.shape

		sign_mat = np.full((PERM, len(self.queries)), -1, dtype='int64')

		count = 0	

		for i in range(PERM):
			perm = np.random.permutation(drows)
			queryList = set()

			partition = np.argsort(perm)

			while len(self.queries) != len(queryList) and len(partition) != 0:
				index_min = partition[0]
				if shingles_matrix[index_min]:
					for q in shingles_matrix[index_min]:
						if not q in queryList:
							queryList.add(q)
							sign_mat[i][q] = perm[index_min]
				partition = np.delete(partition, 0)

			count += 1

			if count % round(PERM * 0.1) == 0:
				print("{} / {} [{}s]".format(count, PERM, round(time.time() - initial, 3)))

		sign_mat = sign_mat.transpose() #get column, i.e. signature

		del shingles_matrix
		gc.collect()

		print(str(round(time.time() - initial, 3)) + "s for signature_matrix")

		return sign_mat

	def compute_querySimilarities(self):

		signatures = self.compute_signatures()
		
		#MAX_CANDIDATES = round(math.log(len(self.queries), 1.5))
		MAX_CANDIDATES = 10

		print("\nMax query candidates: {}, Total queries: {}".format(MAX_CANDIDATES, len(self.queries)))

		# ================================= LSH =================================

		initial = time.time()
		
		lsh = LSH(BAND)

		for sig in signatures:
			lsh.compute_buckets(sig)

		candidates = lsh.get_candidates(signatures)

		print("Candidates pair: {}".format(len(candidates)))

		# ================ COSINE SIMILARITY BETWEEN CANDIDATES =================

		initial = time.time()

		query_sim = {}
		available_query = [False] * len(self.queries)

		for i, j in candidates:
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

		#query_sim = dict(collections.OrderedDict(sorted(query_sim.items())))

		for i in query_sim.keys():
			ind = np.argsort(query_sim[i]["values"])[::-1][0:MAX_CANDIDATES]

			query_sim[i]["indexes"] = np.array(query_sim[i]["indexes"])[ind]
			query_sim[i]["values"] = np.array(query_sim[i]["values"])[ind]

		print(str(round(time.time() - initial, 3)) + "s for queries_similarity scores")

		del signatures
		gc.collect()

		return query_sim, available_query

	def compute_userSimilarities(self):

		#MAX_CANDIDATES = round(math.log(len(self.usersIDs), 1.5))
		MAX_CANDIDATES = 10
		CLUSTER_COUNT = round(len(self.usersIDs) ** (1 / 1.75))

		print("\nMax user candidates: {}, Total users: {}".format(MAX_CANDIDATES, len(self.usersIDs)))

		# ============================== DATA PROCESSING ==============================

		initial = time.time()
			
		scores = np.array(self.ratings)

		for s in range(len(scores)):
			mean = np.mean(scores[s][scores[s] != 0])
			scores[s][scores[s] == 0] = mean

		scaler = StandardScaler()
		normScores = scaler.fit_transform(scores)

		r, c = normScores.shape
		n_comps = min(c, 100) # common heuristic for SVD and PCA

		normScores = PCA(n_components = n_comps).fit_transform(normScores)

		print(str(round(time.time() - initial, 3)) + "s for normalization and PCA")


		# ================================== CLUSTERING ==================================

		print("\nCluster count: {}, Total users: {}".format(CLUSTER_COUNT, len(self.usersIDs)))
		
		initial = time.time()
		
		clusters = Birch(n_clusters = CLUSTER_COUNT).fit(normScores).predict(normScores)

		print(str(round(time.time() - initial, 3)) + "s for clustering")


		# ====================== CENTERED COSINE SIMILARITY BETWEEN CLUSTER ======================

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

		print(str(round(time.time() - initial, 3)) + "s for users_similarity scores")

		return user_sim, top_users, clusters

	def compute_scores(self):

		print("\nQUERY SIMILARITY")
		querySimilarities, available_query = self.compute_querySimilarities()
		
		print("\nUSER SIMILARITY")
		userSimilarities, topUserIndexes, clusters = self.compute_userSimilarities()

		print("\nWEIGHTED AVERAGES")
		scores_to_predict = np.array(np.where(self.ratings == 0)).transpose()

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
			if userPrediction == 0 and queryPrediction == 0:
				self.ratings[i][j] = 0 #cannot find a predictable value
			elif userPrediction == 0:
				self.ratings[i][j] = round(queryPrediction * (QUERY_WEIGHT + (USER_WEIGHT*0.5)) + DEFAULT_MEAN * (USER_WEIGHT*0.5))
			elif queryPrediction == 0:
				self.ratings[i][j] = round(userPrediction * (USER_WEIGHT + (QUERY_WEIGHT*0.5)) + DEFAULT_MEAN * (QUERY_WEIGHT*0.5))
			else:
				self.ratings[i][j] = round(queryPrediction * QUERY_WEIGHT + userPrediction * USER_WEIGHT)

			count += 1

			if count % round(len(scores_to_predict) * 0.1) == 0:
				print("{} / {} [{}s]".format(count, len(scores_to_predict), round(time.time() - initial, 3)))
			

		print(str(round(time.time() - initial, 3)) + "s for weighted averages")

		finalPredictions = pd.DataFrame(self.ratings, columns = self.queriesIDs, index = self.usersIDs)
		scores_missed = np.array(np.where(finalPredictions == 0)).transpose()

		return scores_to_predict, finalPredictions, scores_missed

	def top_k_queries(self, to_predict, predictions, missed):

		command = ""
		user = ""
		k = ""

		predictions = predictions.to_numpy()

		while command != "no":
			while True:
				user = input("Enter user ID: [int][Max: " + str(len(self.usersIDs)) + "] ")
				if user.isdigit():
					user = int(user) - 1
					if user >= 0 and user < len(self.usersIDs):
						break

			just_scored = [j for i, j in to_predict if i == user]

			while True:
				k = input("Enter number of recommendations: [int][Max: " + str(len(just_scored)) + "] ")
				if k.isdigit():
					k = int(k)
					if k > 0 and k <= len(just_scored):
						break

			top_k_predictions = np.argsort(predictions[user][just_scored])[::-1][0:k]

			print("\nTop {} recommendations for U{}: ". format(k, user+1))
			for i in range(len(top_k_predictions)):
				print("{}. Q{} - {}".format(i+1, just_scored[top_k_predictions[i]] + 1, predictions[user][just_scored][top_k_predictions[i]]))
			print()

			command = ""
			while not command.lower() in ["yes", "no"]:
				command = input("Do you want more suggestions? [Yes-No][Default: Yes] ")
				if command == "":
					command = "yes"




	# PART B of ASSIGNMENT [Theorical]

	def suggest_queries(self, predictions):
		
		predictions = predictions.to_numpy().transpose()
		suggestions = []

		initial = time.time()
		for q in range(len(predictions)):
			suggestions.append((self.queriesIDs[q], np.nanmean(predictions[q]), self.tupleCount[q]))

		suggestions.sort(key=lambda item: (item[1], item[2]), reverse=True)

		print(str(round(time.time() - initial, 3)) + "s for queries suggestion")

		return suggestions




	    
	   
	    