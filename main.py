from recommender import Recommender
from resources import generator
#from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import pandas as pd
import math, time
import numpy as np

pd.options.mode.chained_assignment = None 

if __name__ == "__main__":

	initial = time.time()
	users = pd.read_csv("./resources/output/users.csv", names = ["id"], header = None, engine="pyarrow", dtype = {'id':'category'})
	print(str(round(time.time() - initial, 3)) + "s to read users")

	initial = time.time()
	queries = generator.parse_queries("./resources/output/queries.csv").astype({'name':'category', 'address':'category', 'occupation':'category'})
	print(str(round(time.time() - initial, 3)) + "s to read queries")

	initial = time.time()
	dataset = pd.read_csv("./resources/output/dataset.csv", names = ["id","name","address","age","occupation"], header = 0, engine="pyarrow", dtype = {'name':'category', 'address':'category', 'occupation':'category'})
	print(str(round(time.time() - initial, 3)) + "s to read dataset")

	initial = time.time()
	ratings = pd.read_csv("./resources/output/utility_matrix.csv", engine="pyarrow")
	print(str(round(time.time() - initial, 3)) + "s to read utility matrix")

	queriesIDs = list(ratings.columns)
	usersIDs = list(ratings.index.values)

	'''print(users)
	print(dataset)
	print(ratings)
	
	print(queriesIDs)
	print(usersIDs)'''

	recommender = Recommender(users, queries, dataset, ratings)
	to_predict, predictions, missed = recommender.compute_scores()

	print("\nINITIAL RATINGS [{} scores to predict]:".format(to_predict))	
	print(ratings)

	print("\nFINAL PREDICTIONS [{} scores missed - {}%]:".format(missed, round(missed / to_predict, 2) * 100))	
	print(predictions)

	'''
	csv_rows = finalPredictions.to_numpy().tolist()
	
	for ind in range(len(list(finalPredictions.index.values))):
		csv_rows[ind].insert(0, finalPredictions.index.values[ind])

	generator.write_csv("output", finalPredictions.columns, csv_rows)
	'''

	exit(0)