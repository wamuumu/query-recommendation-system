import random
import numpy as np
import pandasql as ps
from itertools import combinations
from scipy.spatial import distance
from scipy import sparse

PERM = 300 #number of independent hash functions (e.g. 100) for computing signatures' matrix

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
		result = ps.sqldf(q1, locals())

		print(result)
		print()

		return result

	def jaccard_similarity(self, list1:list, list2:list):
	    s1 = set(list1)
	    s2 = set(list2)
	    return float(len(s1.intersection(s2)) / len(s1.union(s2)))

	def cosine_similarity(self, list1:list, list2:list):
		return 1 - distance.cosine(list1, list2)

	def euclidean_similarity(self, list1:list, list2:list):
		return 1 - distance.euclidean(list1, list2)

	def centered_cosine_similarity(self, list1:list, list2:list):
		
		mean = np.nanmean(list1)
		list1 = np.array([x if not np.isnan(x) else mean for x in list1 ])
		list1 = list(map(lambda x: x - mean, list1))

		mean = np.nanmean(list2)
		list2 = np.array([x if not np.isnan(x) else mean for x in list2 ])
		list2 = list(map(lambda x: x - mean, list2))

		return self.cosine_similarity(list1, list2)

	def compute_shingles(self):

		qr = self.queries
		rt = self.ratings
		dt = self.dataset

		shingles_matrix = []
		#print(dt)
		for drow in dt.iterrows(): #dataset data
			#print(drow[1]['id'])
			
			mrow = []
			for qrow in rt.columns: #queries of utility matrix
				
				contain = True

				for attr in qr[qrow]:
					tmp = attr.replace('"', '').split("=")
					col, val = tmp[0], tmp[1]

					#print(col, val)

					if str(drow[1][col]) != str(val):
						contain = False
						break

				#print(contain)

				if contain:
					mrow.append(1)
				else:
					mrow.append(0)

			shingles_matrix.append(mrow)

		#print(np.array(shingles_matrix))
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
			col = set()

			partition = np.argpartition(perm, pre).tolist() #find all indexes of argmins in permutation

			for d in range(len(dt)):
				index_min = partition[0]
				for j in range(len(rt.columns)): #iterate over all utility matrix queries
					if j not in col and shingles_matrix[index_min][j] == 1:
						col.add(j)
						signature[j] = perm[index_min]
				partition.remove(partition[0])

				if not -1 in signature:
					break

			sign_mat.append(signature)

		#print(sign_mat)

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

	def compute_querySimilarities(self):

		b, r, thresh = 150, 2, 0.5

		lsh = LSH(b)

		signatures = self.compute_signatures()
		#print(signatures)

		for sign in signatures:
			lsh.add_hash(sign)
		
		#print(lsh.buckets)

		candidate_pairs = lsh.check_candidates()
		print(candidate_pairs)
		
		mat = np.empty((len(signatures), len(signatures)))
		mat[:] = np.nan

		for i, j in candidate_pairs:
			sim = self.cosine_similarity(signatures[i], signatures[j])
			mat[i][j] = sim
			mat[j][i] = sim
			if sim >= thresh:
				print("[" + str(i) + ", " + str(j) + "] -> ok (" + str(sim) + ") >= " + str(thresh) + ")")
			else:
				print("[" + str(i) + ", " + str(j) + "] -> no (" + str(sim) + ") < " + str(thresh) + ")")

		return mat

	def compute_userSimilarities(self):

		users = list(self.ratings.index.values)

		mat = np.zeros((len(users), len(users)))

		for i in range(len(users)):
			for j in range(len(users)):
				#print(self.ratings.to_numpy()[i], self.ratings.to_numpy()[j])
				#print()
				mat[i][j] = self.centered_cosine_similarity(self.ratings.to_numpy()[i], self.ratings.to_numpy()[j])

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




	    
	   
	    