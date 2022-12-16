from recommender import Recommender
from resources import generator
from datatable import dt
#from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import time
import gc


if __name__ == "__main__":

	# Fetch initial data 

	initial = time.time()
	users = dt.fread("./resources/output/users.csv", header=False)
	print(str(round(time.time() - initial, 3)) + "s to read users")

	initial = time.time()
	queries, queriesIDs = generator.parse_queries("./resources/output/queries.csv")
	print(str(round(time.time() - initial, 3)) + "s to read queries")

	initial = time.time()
	dataset = dt.fread("./resources/output/dataset.csv")
	print(str(round(time.time() - initial, 3)) + "s to read dataset")

	initial = time.time()
	cols = ["user"] + queriesIDs 
	ratings = dt.fread("./resources/output/utility_matrix.csv", columns=cols)
	print(str(round(time.time() - initial, 3)) + "s to read utility matrix")

	# Reccomender class to predict values
	recommender = Recommender(users, queries, queriesIDs, dataset, ratings)

	del users, queries, dataset, ratings
	gc.collect()

	# ----------- PART A ------------
	print("\nPART A\n")
	to_predict, predictions, missed = recommender.compute_scores()

	print("\nFINAL PREDICTIONS [{} scores to predict, {} scores missed - {}% miss]:".format(to_predict, missed, round(missed / to_predict, 3) * 100))	
	print(predictions)

	# Save prediction in csv file using generator csv writer
	
	csv_rows = predictions.to_numpy().tolist()
	
	for ind in range(len(list(predictions.index.values))):
		csv_rows[ind].insert(0, predictions.index.values[ind])

	generator.write_csv("output", predictions.columns, csv_rows)


	# ----------- PART B ------------
	'''
	print("\nPART B\n")
	suggestions = recommender.suggest_queries(predictions)

	print("\nSUGGESTIONS:")
	print(suggestions)
	'''

	exit(0)


