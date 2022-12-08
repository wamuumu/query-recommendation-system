from itertools import combinations

import numpy as np

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