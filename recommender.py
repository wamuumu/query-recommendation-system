import random
import numpy as np
import pandasql as ps
from itertools import combinations
from scipy.spatial import distance
from scipy import sparse
import time

PERM = 100 #number of independent hash functions (e.g. 100) for computing signatures' matrix
b = 50
QUERY_THRESH = 0.35
USER_THRESH = 0.35
features = ["name","address","age","occupation"]

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
		#print(q1)
		result = ps.sqldf(q1, locals())

		#print(result)
		#print()

		return result

	def jaccard_similarity(self, list1:list, list2:list):
	    s1 = set(list1)
	    s2 = set(list2)
	    return float(len(s1.intersection(s2)) / len(s1.union(s2)))

	def cosine_similarity(self, list1:list, list2:list):
		return 1 - distance.cosine(list1, list2)

	def euclidean_similarity(self, list1:list, list2:list):
		return 1 - distance.euclidean(list1, list2)

	def compute_shingles(self):

		initial = time.time()

		queries = self.queries.to_numpy()
		users = self.dataset.drop(columns=['id'])

		shingles_matrix = np.zeros((len(users), len(queries)))
		
		#print(queries)
		#print(users)
		#print(features)

		for q in range(len(queries)):
			filteredDataset = users.loc[((queries[q][0] == '') or (users[features[0]] == queries[q][0])) & ((queries[q][1] == '') or (users[features[1]] == queries[q][1])) & ((queries[q][2] == '') or (users[features[2]] == int(queries[q][2]))) & ((queries[q][3] == '') or (users[features[3]] == queries[q][3]))]

			for ind in filteredDataset.index:
				shingles_matrix[ind][q] = 1

			print(str((q / len(queries))*100) + "%")

		print(str(time.time() - initial) + "s for shingles_matrix")
		return shingles_matrix

	def compute_signatures(self):
		
		initial = time.time()
		shingles_matrix = self.compute_shingles()

		sign_mat = []
		perm = [i for i in range(0, len(self.dataset))] #inital permutation (serie)
		pre = perm #save initial indexes

		for i in range(PERM):
			signature = [-1] * len(self.ratings.columns)
			random.shuffle(perm) #shuffle the indexes

			'''
			partition = np.argpartition(perm, pre).tolist() #find all indexes of argmins in permutation
			countQuery = 0
			for d in range(len(perm)):
				index_min = partition[0]
				for j in range(len(self.ratings.columns)): #iterate over all utility matrix queries
					if not pickedQuery[j] and shingles_matrix[index_min][j] == 1:
						pickedQuery[j] = True
						countQuery += 1
						signature[j] = perm[index_min]
				partition.remove(partition[0])

				if countQuery == len(signature):
					break

			'''
			for q in range(len(self.ratings.columns)):
				for d in range(len(perm)):
					if shingles_matrix[perm[d]][q] == 1:
						signature[q] = perm[d]
						break
			sign_mat.append(signature)

			print(str((len(sign_mat) / PERM)*100) + "%")

		print(str(time.time() - initial) + "s for signature_matrix")
		return np.array(sign_mat).transpose() #get column, i.e. signature

	def compute_querySimilarities(self):

		lsh = LSH(b)

		signatures = self.compute_signatures()

		initial = time.time()

		for sign in signatures:
			lsh.add_hash(sign)
		
		#print(lsh.buckets)

		candidate_pairs = lsh.check_candidates()
		print(candidate_pairs)
		
		mat = np.empty((len(signatures), len(signatures)))
		mat[:] = 0

		for i, j in candidate_pairs:
			sim = self.cosine_similarity(signatures[i], signatures[j])
			if sim >= QUERY_THRESH:
				mat[i][j] = sim
				mat[j][i] = sim
			else:
				mat[i][j] = 0
				mat[j][i] = 0

		print(str(time.time() - initial) + "s for candidate_pairs")
		return mat

	def compute_userSimilarities(self):

		users = list(self.ratings.index.values)
		norm = self.ratings.copy().to_numpy()

		mat = np.zeros((len(users), len(users)))

		#normalize rating's vector for pearson's similarity
		for i in range(len(users)):		
			mean = np.nanmean(norm[i])
			norm[i] = np.where(np.isnan(norm[i]), 0, norm[i] - mean)

		initial = time.time()
		for i in range(len(users)):
			for j in range(i+1, len(users)):
				sim = self.cosine_similarity(norm[i], norm[j])
				if sim >= USER_THRESH:
					mat[i][j] = sim
					mat[j][i] = sim
				else:
					mat[i][j] = 0
					mat[j][i] = 0
			print(str(((i+1) / len(users))*100) + "%")

		print(str(time.time() - initial) + "s for users_similarity")
		return mat

class LSH:

	buckets = []
	counter = 0

	def __init__(self, b):
		self.b = b
		for i in range(b):
			self.buckets.append({})

	def make_subvecs(self, signature):

		l = len(signature)
		assert l % self.b == 0
		r = int(l / self.b)
		
		# break signature into subvectors
		subvecs = []
		for i in range(0, l, r):
			subvecs.append(signature[i:i+r])

		return np.stack(subvecs)

	def jaccard_similarity(self, list1:list, list2:list):
	    s1 = set(list1)
	    s2 = set(list2)
	    return float(len(s1.intersection(s2)) / len(s1.union(s2)))

	def cosine_similarity(self, list1:list, list2:list):
		return 1 - distance.cosine(list1, list2)

	def add_hash(self, signature):
		#print(signature)
		#print()
		subvecs = self.make_subvecs(signature).astype(str)
		for i, subvec in enumerate(subvecs):
			found = False
			subvec = ','.join(subvec)
			#print(i, subvec)
			if len(self.buckets[i].keys()) > 0:
				found = False
				for k in self.buckets[i].keys():
					if subvec == k:
						self.buckets[i][k].append(self.counter)
						found = True

				if not found:
					self.buckets[i][subvec] = []
					self.buckets[i][subvec].append(self.counter)
			else:
				self.buckets[i][subvec] = []
				self.buckets[i][subvec].append(self.counter)
		self.counter += 1

	def check_candidates(self):
		candidates = []
		for bucket_band in self.buckets:
			keys = bucket_band.keys()
			for bucket in keys:
				hits = bucket_band[bucket]
				if len(hits) > 1:
					candidates.extend(combinations(hits, 2))
		return set(candidates)




	    
	   
	    