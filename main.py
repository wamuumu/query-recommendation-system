import pandas as pd
import pandasql as ps
import random
import numpy as np
from scipy.spatial import distance

class Recommender:

	def __init__(self, users, queries, dataset, ratings):
		self.users = users
		self.queries = queries
		self.dataset = dataset
		self.ratings = ratings

	def compute_query(self, attributes:list):
		df = self.dataset

		attr = " and ".join(attributes)
		query = "select * from df where " + attr
		
		#print("Query: " + query)

		return ps.sqldf(query) 

	def jaccard_similarity(self, list1:list, list2:list):
	    s1 = set(list1)
	    s2 = set(list2)
	    return float(len(s1.intersection(s2)) / len(s1.union(s2)))

	def cosine_similarity(self, list1:list, list2:list):
		return 1 - distance.cosine(list1, list2)

	def compute_query_similarity(self): #queries' indexes
		qr = self.queries

		matrix = []
		for drow in self.dataset.iterrows(): #people
			#print(drow[1]['id'])
			mrow = []
			for qrow in self.ratings.columns: #queries of utility matrix
				
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

			matrix.append(mrow)

		#print(np.array(matrix))
		
		PERM = 10 #number of independent hash functions (e.g. 100)

		sign_mat = []
		perm = [i for i in range(0, len(self.dataset))]

		for i in range(PERM):
			signature = [-1 for z in range(len(self.ratings.columns))]
			pre = perm
			random.shuffle(perm) #shuffle the indexes

			partition = np.argpartition(perm, pre).tolist()

			for d in range(len(self.dataset)):
				index_min = partition[0]
				for j in range(len(self.ratings.columns)): #iterate over all utility matrix queries
					if matrix[index_min][j] == 1:
						signature[j] = perm[index_min]
				partition.remove(partition[0])

				if not -1 in signature:
					break

			sign_mat.append(signature)

		print(sign_mat)
		
		#SBAGLIATO
		data = [tuple(row) for row in np.array(sign_mat)]
		unique_signatures = np.unique(data, axis=0)
		print(unique_signatures)

		signatures = unique_signatures.transpose() #queries

		print(signatures)
		print()

		for i in range(len(signatures)):
			for j in range(i+1, len(signatures)):
				print(list(ratings.columns)[i] + " - " + list(ratings.columns)[j] + " -> " + "Jaccard: " + str(self.jaccard_similarity(signatures[i], signatures[j])))
			print()

def parse_queries(path:str):

	data = {}

	with open(path) as f:
		for row in f:
			row = row.rstrip('\n')
			values = row.split(",")
			index = values[0]
			values = values[1::]

			data[index] = []
			for attr in values:
				val = attr.split("=")
				if val[0] != "age" and val[0] != "id":
					val[1] = '"' + val[1] + '"'
				val = val[0] + "=" + val[1]
				data[index].append(val)

	return data

if __name__ == "__main__":
		
	users = pd.read_csv("./resources/output/users.csv", names = ["id"], header = None)
	queries = parse_queries("./resources/output/queries.csv")
	dataset = pd.read_csv("./resources/output/dataset.csv", names = ["id","name","address","age","occupation"], header = 0)
	ratings = pd.read_csv("./resources/output/utility_matrix.csv")

	queriesIDs = list(ratings.columns)
	usersIDs = list(ratings.index.values)

	'''print(users)
	print(dataset)
	print(ratings)
	
	print(queriesIDs)
	print(usersIDs)'''

	recommender = Recommender(users, queries, dataset, ratings)
	#result = recommender.compute_query(queries['Q1']) #lista campi in conjunction

	recommender.compute_query_similarity()

	exit(0)