import random
import numpy as np
import pandasql as ps
from itertools import combinations
from scipy.spatial import distance

PERM = 100 #number of independent hash functions (e.g. 100) for computing signatures' matrix

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
		print(q1)

		print(ps.sqldf(q1, locals()))

	def jaccard_similarity(self, list1:list, list2:list):
	    s1 = set(list1)
	    s2 = set(list2)
	    return float(len(s1.intersection(s2)) / len(s1.union(s2)))

	def cosine_similarity(self, list1:list, list2:list):
		return 1 - distance.cosine(list1, list2)

	def euclidean_similarity(self, list1:list, list2:list):
		return 1 - distance.euclidean(list1, list2)

	def compute_shingles(self):

		qr = self.queries
		rt = self.ratings
		dt = self.dataset

		shingles_matrix = []
		for drow in dt.iterrows(): #dataset data
			#print(drow[1]['id'])
			mrow = []
			for qrow in rt.columns: #queries of utility matrix
				
				contain = True

				for attr in qr[qrow]:
					tmp = attr.replace('"', '').split("=")
					col, val = tmp[0], tmp[1]

					if drow[1][col] != val:
						contain = False
						break

				if contain:
					mrow.append(1)
				else:
					mrow.append(0)

			shingles_matrix.append(mrow)

		#print(np.array(matrix))
		return shingles_matrix

	def compute_signatures(self):
		
		qr = self.queries
		rt = self.ratings
		dt = self.dataset

		shingles_matrix = self.compute_shingles()

		# PROBLEMA: con molte permutazioni, le signature tendono ad assomigliarsi sebbene le 
		# queries siano diverse. 

		# Quindi LSH e bucket oppure cambiare metodo di similarità

		'''
	
		A major disadvantage of the Jaccard index is that it is highly influenced by the size of the data. 
		Large datasets can have a big impact on the index as it could significantly increase the union whilst keeping the intersection similar.

		'''

		sign_mat = []
		perm = [i for i in range(0, len(dt))] #inital permutation (serie)
		pre = perm #save initial indexes

		for i in range(PERM):
			signature = [-1] * len(rt.columns)
			random.shuffle(perm) #shuffle the indexes

			partition = np.argpartition(perm, pre).tolist() #find all indexes of argmins in permutation

			for d in range(len(dt)):
				index_min = partition[0]
				for j in range(len(rt.columns)): #iterate over all utility matrix queries
					if shingles_matrix[index_min][j] == 1:
						signature[j] = perm[index_min]
				partition.remove(partition[0])

				if not -1 in signature:
					break

			sign_mat.append(signature)

		# check if signatures are unique
		'''data = [tuple(row) for row in np.array(sign_mat)]

		unique_signatures = np.unique(data, axis=0, return_index=True)[1]
		unique_signatures = np.array([list(data[index]) for index in sorted(unique_signatures)])'''

		#signatures = np.array(sign_mat).transpose() #queries

		'''print(signatures)
		print()

		for i in range(len(signatures)):
			for j in range(len(signatures)):
				print(list(ratings.columns)[i] + " - " + list(ratings.columns)[j] + " -> " + "Jaccard: " + str(self.jaccard_similarity(signatures[i], signatures[j])))
			print()
		'''

		return np.array(sign_mat).transpose() #get column, i.e. signature

	def compute_lsh(self, signatures:list):

		b = 10

		lsh = LSH(b)

		for sign in signatures:
			lsh.add_hash(sign)
		
		candidate_pairs = lsh.check_candidates()
		print(candidate_pairs)

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
				for k in self.buckets[i].keys():
					if self.jaccard_similarity(subvec, k) >= 0.8:
						self.buckets[i][k].append(self.counter)
			
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




	    
	   
	    