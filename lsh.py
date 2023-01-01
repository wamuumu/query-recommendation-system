import numpy as np
from itertools import combinations
import random
import math
import time

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

		return np.stack(subvecs).astype('int16')


	def compute_buckets(self, signature):
		subvecs = self.make_subvecs(signature).astype(str)
		for i, subvec in enumerate(subvecs):
			subvec = ",".join(subvec)
			if not subvec in self.buckets[i]:
				self.buckets[i][subvec] = []
			self.buckets[i][subvec].append(self.counter)
		self.counter += 1

	def get_candidates(self, signatures):
		candidates = set()
		maxval = 0
		maxlen = 0
		count = 0
		initial = time.time()
		for bucket_band in self.buckets:
			for bucket in bucket_band.keys():
				keySet = set(bucket.split(","))
				hits = bucket_band[bucket]

				if len(hits) > 1 and keySet != {'-1'}: #check if band != -1, which means signature is empty

					maxlen = max(maxlen, len(hits))
					comb_iter = list(combinations(hits, 2))
					maxval = max(maxval, len(comb_iter))
					
					for c in comb_iter:
						if not reversed(c) in candidates:
							candidates.add(c)
			
			count += 1
			#print("{} / {} [{}s] - MaxLen of buckets: {}, Max combinations: {}".format(count, len(self.buckets), round(time.time() - initial, 3), maxlen, maxval))

		return candidates