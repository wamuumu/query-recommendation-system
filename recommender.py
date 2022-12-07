from lsh import LSH
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance
from scipy.sparse import csr_matrix

import numpy as np
import matplotlib.pyplot as plt
import collections
import time
import random

PERM = 100 #number of independent hash functions (e.g. 100) for computing signatures' matrix
QUERY_THRESH = 0.65
USER_THRESH = 0.55

features = ["name","address","age","occupation"]

random.seed(time.time())

class Recommender:

	def __init__(self, users, queries, dataset, ratings):
		self.users = users
		self.queries = queries.to_numpy()
		self.dataset = dataset
		self.ratings = ratings

	def compute_shingles(self):

		initial = time.time()

		shingles_matrix = {}

		for q in range(len(self.queries)):

			filteredDataset = self.dataset
			for f in range(len(features)):
				if self.queries[q][f] != "":
					if self.queries[q][f].isdigit():
						filteredDataset = filteredDataset.loc[filteredDataset[features[f]] == int(self.queries[q][f])]
					else:
						filteredDataset = filteredDataset.loc[filteredDataset[features[f]] == self.queries[q][f]]

			for ind in filteredDataset.index.values:
				if not ind in shingles_matrix.keys():
					shingles_matrix[ind] = []
				shingles_matrix[ind].append(q)
	
			print(str((q / len(self.queries))*100) + "%")

		print(str(time.time() - initial) + "s for shingles_matrix")
		return shingles_matrix

	def compute_signatures(self):
		
		shingles_matrix = self.compute_shingles()

		initial = time.time()

		sign_mat = np.empty((PERM, len(self.queries)))
		sign_mat[:] = -1
		queryList = set()

		perm = [d for d in range(len(self.dataset))]

		for i in range(PERM):
			random.shuffle(perm) #shuffle the indexes

			partition = np.argsort(perm)

			while len(self.queries) != len(queryList) and len(partition) != 0:
				index_min = partition[0]
				if index_min in shingles_matrix.keys():
					for q in shingles_matrix[index_min]:
						if not q in queryList:
							queryList.add(q)
							sign_mat[i][q] = perm[index_min]
				partition = np.delete(partition, 0)

			print(str(((i+1) / PERM)*100) + "%")

		print(str(time.time() - initial) + "s for signature_matrix")
		return sign_mat.transpose() #get column, i.e. signature

	def compute_querySimilarities(self):

		signatures = self.compute_signatures()

		initial = time.time()
		
		
		# SPARSE MATRIX APPROACH 43.6s
		sig_sparse = csr_matrix(signatures)
		sig_sim = cosine_similarity(sig_sparse)

		for i in range(len(signatures)):
			sig_sim[i][i] = 0

			emptySig = False
			if all(s == -1 for s in signatures[i]):
				emptySig = True

			for j in range(i+1, len(signatures)):
				if sig_sim[i][j] < QUERY_THRESH or emptySig:
					sig_sim[i][j] = 0
					sig_sim[j][i] = 0

			print(str(((i+1) / len(signatures))*100) + "%")

		'''
		# LSH APPROACH USING BUCKETS (10000 data - 10000 query - 1000 user -> Memory error)
		
		QUERY_THRESH = 0.35
		USER_THRESH = 0.35
		buckets = 25 #bands of size PERM / buckets

		lsh = LSH(buckets)

		for sig in signatures:
			lsh.add_hash(sig)
		
		#print(lsh.buckets)

		candidate_pairs = lsh.check_candidates()
		#print(candidate_pairs)
		
		sig_sim = np.empty((len(signatures), len(signatures)))
		sig_sim[:] = 0

		count = 0
		for i, j in candidate_pairs:
			sim = cosine_similarity([signatures[i]], [signatures[j]])[0][0]
			if sim >= QUERY_THRESH:
				sig_sim[i][j] = sim
				sig_sim[j][i] = sim
			else:
				sig_sim[i][j] = 0
				sig_sim[j][i] = 0

			count += 1

			print(str((count / len(candidate_pairs)) * 100) + "%")

		'''
		print(str(time.time() - initial) + "s for queries_similarity")
		return sig_sim

	def compute_userSimilarities(self):

		initial = time.time()

		norm = self.ratings.copy().fillna(0).to_numpy()
		user_sim = np.corrcoef(norm)

		# 43s
		for i in range(len(self.ratings.index.values)):
			user_sim[i][i] = 0
			for j in range(i+1, len(self.ratings.index.values)):
				if user_sim[i][j] < USER_THRESH:
					user_sim[i][j] = 0
					user_sim[j][i] = 0
			print(str(((i+1) / len(self.ratings.index.values))*100) + "%")

		print(str(time.time() - initial) + "s for users_similarity")
		
		return user_sim




	    
	   
	    