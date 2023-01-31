# query similarity
from lsh import LSH

# clustering 
from sklearn.cluster import Birch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# mathematical computations
from sklearn.metrics.pairwise import cosine_similarity

# dataframes to handle csv
from datatable import dt, f, ifelse, update
import pandas as pd

# general imports
import numpy as np
import itertools
import collections
import math
import time

# numba
import numba
from numba import jit

pd.options.mode.chained_assignment = None 

# constants
PERM = 180 # number of independent hash functions

QUERY_WEIGHT = 0.6
USER_WEIGHT = 0.4
DEFAULT_MEAN = 60

@jit(nopython=True)
def weighted_average(ratings, indexes, simScores):

	prediction = 0
	userRating = ratings[indexes]

	weightSum = np.sum(simScores[userRating != 0])

	if weightSum != 0:
		prediction = np.sum(userRating * simScores) / weightSum

	return prediction

class Recommender:

	def init(self, users, queries, queriesIDs, dataset, ratings):
		self.usersIDs = users.to_numpy().T[0]

		self.queries = queries.to_numpy()
		self.queriesIDs = np.array(queriesIDs)
		
		dataset[:] = dt.str64
		self.dataset = dataset.to_pandas()
		self.tupleCount = {}
		
		del ratings[:, ['user']] #delete user from ratings
		ratings[:, update(**{key: ifelse(f[key] == None, 0, f[key]) for key in ratings.names})] #replace NaN with 0

		self.ratings = ratings.to_numpy()

	# MAIN COMPUTATIONAL METHODS

	def compute_shingles(self):

		drows, dcols = self.dataset.shape

		print("\nDataset : {}, Total queries: {}".format(drows, self.queriesIDs.size))

		initial = time.time()

		shingles_dict = {}

		for d in range(drows):
			shingles_dict[d] = []

		count = 0

		for q in range(self.queriesIDs.size):

			cond = True
			for ft in range(len(self.datasetFeatures)):
				if self.queries[q][ft] != "":
					cond &= (self.dataset[self.datasetFeatures[ft]] == self.queries[q][ft])

			answer_set = cond[cond].index.values

			self.tupleCount[q] = len(answer_set)

			for ind in answer_set:
				shingles_dict[ind].append(q)

			count += 1

			if count % max(1, round(self.queriesIDs.size * 0.1)) == 0:
				print("{} / {} [{}s]".format(count, self.queriesIDs.size, round(time.time() - initial, 3)))

		print(str(round(time.time() - initial, 3)) + "s for shingles_dict")
		return shingles_dict

	def compute_signatures(self):
		
		shingles_dict = self.compute_shingles()
		drows, dcols = self.dataset.shape

		print("\nPermutations: {}".format(PERM))

		initial = time.time()

		sign_mat = np.full((PERM, self.queriesIDs.size), -1, dtype=int)

		count = 0

		for i in range(PERM):	

			perm = np.random.permutation(drows) # random permutation 3 1 2
			sorted_indexes = np.argsort(perm) # order of the indexes 1 2 0
			
			query_set = set()

			for ind in sorted_indexes:
				for q in shingles_dict[ind]:
					if not q in query_set:
						query_set.add(q)
						sign_mat[i][q] = perm[ind]

				if len(query_set) == self.queries.size:
					break

			count += 1

			if count % max(1, round(PERM * 0.1)) == 0:
				print("{} / {} [{}s]".format(count, PERM, round(time.time() - initial, 3)))

		sign_mat = sign_mat.T # get columns, i.e. signatures

		print(str(round(time.time() - initial, 3)) + "s for signature_matrix")
		
		return sign_mat

	def compute_querySimilarities(self):

		queryTime = time.time()

		signatures = self.compute_signatures()
		
		MAX_CANDIDATES = round(math.log(self.queriesIDs.size, 1.5))

		LSH_THRESH = 0.2 # lower this value [0...1] to find more candidates with less precision

		print("\nQuery Thresh: " + str(LSH_THRESH))
		for b in list(range(1, PERM+1))[::-1]:
			if PERM % b == 0 and b % 10 == 0:
				r = PERM / b
				thresh = round((1/b) ** (1/r), 2)
				print(b, r, thresh)
				if thresh >= LSH_THRESH:
					band = b
					break

		print("\nMax query candidates: {}, Max bands: {}, Band size: {}, Total queries: {}".format(MAX_CANDIDATES, band, PERM / band, self.queriesIDs.size))

		# ================================= LSH =================================

		initial = time.time()

		lsh = LSH(band)

		for sig in signatures:
			lsh.compute_buckets(sig)

		candidates = lsh.get_candidates(signatures)

		print("Candidate pairs [{}s]: {}".format(round(time.time() - initial, 3), len(candidates)))

		# ================ COSINE SIMILARITY BETWEEN CANDIDATES =================

		initial = time.time()
		query_sim = {}

		for i, j in candidates:
			if not i in query_sim:
				query_sim[i] = {}
				query_sim[i]['indexes'] = []
				query_sim[i]['signatures'] = []
				query_sim[i]['signatures'].append(signatures[i])

			if not j in query_sim:
				query_sim[j] = {}
				query_sim[j]['indexes'] = []
				query_sim[j]['signatures'] = []
				query_sim[j]['signatures'].append(signatures[j])

			query_sim[i]['indexes'].append(j)
			query_sim[j]['indexes'].append(i)
			query_sim[i]['signatures'].append(signatures[j])
			query_sim[j]['signatures'].append(signatures[i])

		for i in query_sim:
			query_sim[i]['values'] = np.around(cosine_similarity(query_sim[i]['signatures'])[0][1::], 3)
			del query_sim[i]['signatures']

			ind = np.argsort(query_sim[i]["values"])[::-1][0:MAX_CANDIDATES]

			query_sim[i]["indexes"] = np.array(query_sim[i]["indexes"])[ind]
			query_sim[i]["values"] = np.array(query_sim[i]["values"])[ind]

		print("\n"+str(round(time.time() - queryTime, 3)) + "s for overall queries_similarity scores")

		return query_sim

	def compute_userSimilarities(self):

		userTime = time.time()

		MAX_CANDIDATES = round(math.log(self.usersIDs.size, 1.5))
		CLUSTER_COUNT = round(self.usersIDs.size ** (1/1.3)) # criteria to stop global clustering of BIRCH

		print("\nMax user candidates: {}, Total users: {}".format(MAX_CANDIDATES, self.usersIDs.size))

		# ============================== DATA PROCESSING ==============================

		initial = time.time()
			
		scaler = StandardScaler()
		normScores = scaler.fit_transform(self.ratings)

		r, c = normScores.shape
		n_comps = min(r, c, 200)

		pca = PCA(n_components = n_comps).fit(normScores)
		normScores = pca.transform(normScores)

		print("{}s for normalization and PCA [{} components]".format(round(time.time() - initial, 3), n_comps))

		# ================================== CLUSTERING ==================================

		print("\nCluster count: {}, Total users: {}".format(CLUSTER_COUNT, self.usersIDs.size))
		
		initial = time.time()
		
		clusters = Birch(n_clusters = CLUSTER_COUNT).fit(normScores).predict(normScores)

		print(str(round(time.time() - initial, 3)) + "s for clustering")

		# ====================== CENTERED COSINE SIMILARITY BETWEEN CLUSTER ======================
		
		counts = np.bincount(clusters)

		for i in range(counts.size):
			if counts[i] == 1:
				clusters[clusters == i] = CLUSTER_COUNT

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

		print("\n"+str(round(time.time() - userTime, 3)) + "s for overall users_similarity scores")

		return user_sim

	def compute_scores(self):

		print("\n========== QUERY SIMILARITY ==========")
		querySimilarities = self.compute_querySimilarities()
		
		print("\n========== USER SIMILARITY ==========")
		userSimilarities = self.compute_userSimilarities()

		print("\n========== WEIGHTED AVERAGES ==========")
		finalPredictions = self.ratings.copy()
		scores_to_predict = np.array(np.where(self.ratings == 0)).T

		queryPrediction = 0
		userPrediction = 0

		initial = time.time()

		count = 0
		
		for i, j in scores_to_predict:

			# CONTENT-BASED FILTERING QUERY-QUERY
			if j in querySimilarities:
				queryPrediction = weighted_average(self.ratings[i], querySimilarities[j]["indexes"], querySimilarities[j]["values"])
			else:
				queryPrediction = 0
			
			# COLLABORATIVE FILTERING USER-USER
			userPrediction = weighted_average(self.ratings.T[j], userSimilarities[i]["indexes"], userSimilarities[i]["values"])


			# HYBRID PREDICTIONS
			if userPrediction == 0 and queryPrediction == 0:
				finalPredictions[i][j] = 0 #cannot find a predictable value
			elif userPrediction == 0:
				finalPredictions[i][j] = round(queryPrediction * (QUERY_WEIGHT + (USER_WEIGHT*0.5)) + DEFAULT_MEAN * (USER_WEIGHT*0.5))
			elif queryPrediction == 0:
				finalPredictions[i][j] = round(userPrediction * (USER_WEIGHT + (QUERY_WEIGHT*0.5)) + DEFAULT_MEAN * (QUERY_WEIGHT*0.5))
			else:
				finalPredictions[i][j] = round(queryPrediction * QUERY_WEIGHT + userPrediction * USER_WEIGHT)

			count += 1

			if count % max(1, round(len(scores_to_predict) * 0.1)) == 0:
				print("{} / {} [{}s]".format(count, len(scores_to_predict), round(time.time() - initial, 3)))		

		print(str(round(time.time() - initial, 3)) + "s for weighted averages")

		finalPredictions = pd.DataFrame(finalPredictions, columns = self.queriesIDs, index = self.usersIDs).astype(int)
		scores_missed = np.array(np.where(finalPredictions == 0)).T

		return scores_to_predict, finalPredictions, scores_missed

	def top_k_queries(self, to_predict, predictions, missed):

		command = ""
		user = ""
		k = ""

		predictions = predictions.to_numpy()

		while command.lower() != "no":
			while True:
				user = input("Enter user ID: [int][Max: " + str(self.usersIDs.size) + "] ")
				if user.isdigit():
					user = int(user) - 1
					if user >= 0 and user < self.usersIDs.size:
						break

			just_scored = [j for i, j in to_predict if i == user and predictions[i][j] != 0]

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


	# OTHER METHODS

	def parse_queries(self, path:str):

		data = []
		indexes = []
		pdict = {}
		lineCount = 0

		with open(path) as f:
			for row in f:
				lineCount += 1
				row = row.rstrip('\n')
				values = row.split(",")
				indexes.append(values[0])
				values = values[1::]

				element = ["" for i in range(len(self.datasetFeatures))]

				for val in values:
					attr = val.split("=") #attr[0] -> feature's name, attr[1] -> feature's value
					ind = self.datasetFeatures.index(attr[0])
					element[ind] = attr[1]

				data.append(element)

		if lineCount > 0:
			data = np.array(data).T

			for i in range(len(self.datasetFeatures)):
				pdict[self.datasetFeatures[i]] = data[i] 

		return dt.Frame(pdict), indexes




	    
	   
	    