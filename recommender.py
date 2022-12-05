from lsh import LSH
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

import pandasql as ps
import numpy as np
import time
import random

PERM = 150 #number of independent hash functions (e.g. 100) for computing signatures' matrix
b = 75
QUERY_THRESH = 0.65
USER_THRESH = 0.55

features = ["name","address","age","occupation"]
empty_queries = []

random.seed(time.time())

class Recommender:

	def __init__(self, users, queries, dataset, ratings):
		self.users = users
		self.queries = queries
		self.dataset = dataset
		self.ratings = ratings

	def compute_query(self, attributes:list):

		df = self.dataset
		attr = " and ".join(attributes)

		q1 = "SELECT * FROM df WHERE " + attr
		result = ps.sqldf(q1, locals())

		return result

	def compute_shingles(self):

		initial = time.time()

		queries = self.queries.to_numpy()
		users = self.dataset.drop(columns=['id'])

		shingles_matrix = np.zeros((len(users), len(queries)))

		for q in range(len(queries)):
			filteredDataset = users.loc[((queries[q][0] == '') or (users[features[0]] == queries[q][0])) & ((queries[q][1] == '') or (users[features[1]] == queries[q][1])) & ((queries[q][2] == '') or (users[features[2]] == int(queries[q][2]))) & ((queries[q][3] == '') or (users[features[3]] == queries[q][3]))]
			
			if filteredDataset.empty:
				empty_queries.append(q)
				continue

			for ind in filteredDataset.index:
				shingles_matrix[ind][q] = 1

			print(str((q / len(queries))*100) + "%")

		print(str(time.time() - initial) + "s for shingles_matrix")
		return shingles_matrix

	def compute_signatures(self):
		
		initial = time.time()
		shingles_matrix = self.compute_shingles()
		shingles_row, shingles_col = shingles_matrix.shape

		sign_mat = []
		one_values = []
		perm = []

		for d in range(shingles_row):
			perm.append(d)
			one_values.append(np.where(shingles_matrix[d] == 1)[0])

		for i in range(PERM):
			signature = [-1] * len(self.ratings.columns)
			random.shuffle(perm) #shuffle the indexes
			pickedQuery = [False] * len(signature)

			partition = np.argsort(perm)
			sig_count = 0

			while True:
				index_min = partition[0]
				for q in one_values[index_min]:
					if not pickedQuery[q]:
						signature[q] = perm[index_min]
						pickedQuery[q] = True
						sig_count += 1
				partition = np.delete(partition, 0)

				if len(signature) == sig_count or len(partition) == 0:
					break

			sign_mat.append(signature)

			print(str((len(sign_mat) / PERM)*100) + "%")

		print(str(time.time() - initial) + "s for signature_matrix")
		return np.array(sign_mat).transpose() #get column, i.e. signature

	def compute_querySimilarities(self):

		signatures = self.compute_signatures()

		initial = time.time()


		# SPARSE MATRIX APPROACH
		sig_sparse = csr_matrix(signatures)
		sig_sim = cosine_similarity(sig_sparse)

		for i in range(len(signatures)):
			sig_sim[i][i] = 0
			for j in range(i+1, len(signatures)):
				if sig_sim[i][j] < QUERY_THRESH:
					sig_sim[i][j] = 0
					sig_sim[j][i] = 0

		'''
		# LSH APPROACH USING BUCKETS

		lsh = LSH(b)

		for sig in signatures:
			lsh.add_hash(sig)
		
		#print(lsh.buckets)

		candidate_pairs = lsh.check_candidates()
		print(candidate_pairs)
		
		sig_sim = np.empty((len(signatures), len(signatures)))
		sig_sim[:] = 0

		for i, j in candidate_pairs:
			sim = cosine_similarity([signatures[i]], [signatures[j]])[0][0]
			if sim >= QUERY_THRESH:
				sig_sim[i][j] = sim
				sig_sim[j][i] = sim
			else:
				sig_sim[i][j] = 0
				sig_sim[j][i] = 0
		'''

		print(str(time.time() - initial) + "s for candidate_pairs")
		return sig_sim

	def compute_userSimilarities(self):

		users = list(self.ratings.index.values)
		norm = self.ratings.copy().to_numpy()

		#normalize rating's vector for pearson's similarity
		for i in range(len(users)):		
			mean = np.nanmean(norm[i])
			norm[i] = np.where(np.isnan(norm[i]), 0, norm[i] - mean)

		initial = time.time()

		# SPARSE MATRIX APPROACH
		norm_sparse = csr_matrix(norm)
		user_sim = cosine_similarity(norm_sparse)

		for i in range(len(users)):
			user_sim[i][i] = 0
			for j in range(i+1, len(users)):
				if user_sim[i][j] < USER_THRESH:
					user_sim[i][j] = 0
					user_sim[j][i] = 0
			print(str(((i+1) / len(users))*100) + "%")

		print(str(time.time() - initial) + "s for users_similarity")

		return user_sim




	    
	   
	    