# query similarity
from lsh import LSH

# clustering 
from sklearn.cluster import Birch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# mathematical computations
from sklearn.metrics.pairwise import cosine_similarity

# dataframes to handle csv
from datatable import dt, f
import pandas as pd

# general imports
import numpy as np
import itertools
import collections
import math
import time

pd.options.mode.chained_assignment = None 

# constants
PERM = 120 # number of independent hash functions (e.g. 100) for computing signatures' matrix
BAND = 40 # number of bands in a signature

QUERY_WEIGHT = 0.6
USER_WEIGHT = 0.4
DEFAULT_MEAN = 60

class Recommender:

	def __init__(self, users, queries, queriesIDs, dataset, ratings):
		self.usersIDs = users.to_numpy().transpose()[0]

		self.queries = queries.to_numpy()
		self.queriesIDs = np.array(queriesIDs)
		
		dataset[:] = dt.str64
		self.datasetFeatures = list(dataset.names)[1::]
		self.dataset = dataset.to_pandas()
		self.tupleCount = {}
		
		ratings.replace({None: 0}) #replace NaN with 0
		del ratings[:, ['user']] 
		self.ratings = ratings.to_numpy()
		self.ratingsT = self.ratings.transpose()

	def compute_shingles(self):

		drows, dcols = self.dataset.shape

		print("\nDataset : {}, Total queries: {}".format(drows, self.queriesIDs.size))

		initial = time.time()

		shingles_dict = {}

		for d in range(drows):
			shingles_dict[d] = []

		count = 0
		for q in range(self.queriesIDs.size):

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
				shingles_dict[ind].append(q)

			count += 1

			if count % round(self.queriesIDs.size * 0.1) == 0:
				print("{} / {} [{}s]".format(count, self.queriesIDs.size, round(time.time() - initial, 3)))

		#shingles_dict = dict(collections.OrderedDict(sorted(shingles_dict.items())))

		print(str(round(time.time() - initial, 3)) + "s for shingles_dict")
		return shingles_dict

	def compute_signatures(self):
		
		shingles_dict = self.compute_shingles()

		print("\nPermutations: {}".format(PERM))

		initial = time.time()

		sign_mat = np.full((PERM, self.queriesIDs.size), -1, dtype='int64')

		count = 0	
		
		for i in range(PERM):
			perm = np.random.permutation(len(shingles_dict))
			queryList = set()

			partition = np.argsort(perm)

			# 25s
			while self.queriesIDs.size != len(queryList) and partition.size != 0:
				index_min = partition[0]
				if shingles_dict[index_min]:
					for q in shingles_dict[index_min]:
						if not q in queryList:
							queryList.add(q)
							sign_mat[i][q] = perm[index_min]
				partition = partition[1::]

			count += 1

			if count % round(PERM * 0.1) == 0:
				print("{} / {} [{}s]".format(count, PERM, round(time.time() - initial, 3)))

		sign_mat = sign_mat.transpose() #get column, i.e. signature

		print(str(round(time.time() - initial, 3)) + "s for signature_matrix")

		return sign_mat

	def compute_querySimilarities(self):

		signatures = self.compute_signatures()
		
		MAX_CANDIDATES = round(math.log(self.queriesIDs.size, 1.5))

		print("\nMax query candidates: {}, Total queries: {}".format(MAX_CANDIDATES, self.queriesIDs.size))

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
		available_query = set()

		for i, j in candidates:
			sim = round(max(0, cosine_similarity([signatures[i], signatures[j]])[0][1]), 3)

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

			available_query.add(i)
			available_query.add(j)

		#query_sim = dict(collections.OrderedDict(sorted(query_sim.items())))

		for i in query_sim:
			ind = np.argsort(query_sim[i]["values"])[::-1][0:MAX_CANDIDATES]

			query_sim[i]["indexes"] = np.array(query_sim[i]["indexes"])[ind]
			query_sim[i]["values"] = np.array(query_sim[i]["values"])[ind]

		print(str(round(time.time() - initial, 3)) + "s for queries_similarity scores")

		return query_sim, available_query

	def compute_userSimilarities(self):

		MAX_CANDIDATES = round(math.log(self.usersIDs.size, 1.5))
		CLUSTER_COUNT = round(self.usersIDs.size ** (1 / 1.75))

		print("\nMax user candidates: {}, Total users: {}".format(MAX_CANDIDATES, self.usersIDs.size))

		# ============================== DATA PROCESSING ==============================

		initial = time.time()
			
		scores = np.array(self.ratings)

		for s in range(len(scores)):
			mean = np.mean(scores[s][scores[s] != 0])
			scores[s][scores[s] == 0] = mean

		scaler = StandardScaler()
		normScores = scaler.fit_transform(scores)

		r, c = normScores.shape
		n_comps = min(r, c, 100) # common heuristic for SVD and PCA

		normScores = PCA(n_components = n_comps).fit_transform(normScores)

		print(str(round(time.time() - initial, 3)) + "s for normalization and PCA")


		# ================================== CLUSTERING ==================================

		print("\nCluster count: {}, Total users: {}".format(CLUSTER_COUNT, self.usersIDs.size))
		
		initial = time.time()
		
		clusters = Birch(n_clusters = CLUSTER_COUNT).fit(normScores).predict(normScores)

		print(str(round(time.time() - initial, 3)) + "s for clustering")


		# ====================== CENTERED COSINE SIMILARITY BETWEEN CLUSTER ======================

		initial = time.time()

		cluster_sim = {}

		for i in clusters:
			if not i in cluster_sim:
				cluster_sim[i] = {}
				c_scores = np.array(self.ratings[clusters == i])

				for s in range(len(c_scores)):
					mean = np.mean(c_scores[s][c_scores[s] != 0])
					c_scores[s][c_scores[s] != 0] = c_scores[s][c_scores[s] != 0] - mean

				cluster_sim[i]["indexes"] = np.where(clusters == i)[0]
				cluster_sim[i]["values"] = np.around(cosine_similarity(c_scores), 3)
				np.fill_diagonal(cluster_sim[i]["values"], 0)
				cluster_sim[i]["values"][cluster_sim[i]["values"] < 0] = 0

		user_sim = {}

		for i in range(self.usersIDs.size):
			user_sim[i] = {}
			user_sim[i]["indexes"] = []
			user_sim[i]["values"] = []

			clusterID = clusters[i]
			userID = np.where(cluster_sim[clusterID]["indexes"] == i)[0][0]
			ind = np.argsort(cluster_sim[clusterID]["values"][userID])[::-1][0:MAX_CANDIDATES]

			user_sim[i]["indexes"] = np.array(cluster_sim[clusterID]["indexes"])[ind]
			user_sim[i]["values"]  = np.array(cluster_sim[clusterID]["values"][userID])[ind]

		print(str(round(time.time() - initial, 3)) + "s for users_similarity scores")

		return user_sim

	def compute_scores(self):

		print("\n========== QUERY SIMILARITY ==========")
		querySimilarities, available_query = self.compute_querySimilarities()
		
		print("\n========== USER SIMILARITY ==========")
		userSimilarities = self.compute_userSimilarities()

		print("\n========== WEIGHTED AVERAGES ==========")
		scores_to_predict = np.array(np.where(self.ratings == 0)).transpose()

		queryPrediction = 0
		userPrediction = 0

		initial = time.time()

		count = 0

		for i, j in scores_to_predict:

			# COLLABORATIVE FILTERING QUERY-QUERY 
			if j in available_query:

				userRating = self.ratings[i][querySimilarities[j]["indexes"]]
				simScores = querySimilarities[j]["values"]

				simScores[userRating == 0] = 0
				weightSum = np.sum(simScores)

				if weightSum == 0:
					queryPrediction = 0
				else:
					queryPrediction = np.sum(userRating * simScores) / weightSum
			else:
				queryPrediction = 0


			# COLLABORATIVE FILTERING USER-USER
			userRating = self.ratingsT[j][userSimilarities[i]["indexes"]]
			simScores = userSimilarities[i]["values"]
			
			simScores[userRating == 0] = 0
			weightSum = np.sum(simScores)

			if weightSum == 0:
				userPrediction = 0
			else:
				userPrediction = np.sum(userRating * simScores) / weightSum


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
				user = input("Enter user ID: [int][Max: " + str(self.usersIDs.size) + "] ")
				if user.isdigit():
					user = int(user) - 1
					if user >= 0 and user < self.usersIDs.size:
						break

			just_scored = [j for i, j in to_predict if i == user]

			while True:
				k = input("Enter number of recommendations: [int][Max: " + str(len(just_scored)) + "] ")
				if k.isdigit():
					k = int(k)
					if k > 0 and k <= len(just_scored):
						break

			top_k_predictions = np.argsort(predictions[user][just_scored])[::-1][0:k]

			print("\nTop {} unrated query recommendations for U{}: ". format(k, user+1))
			for i in range(top_k_predictions.size):
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




	    
	   
	    