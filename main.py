from recommender import Recommender
from resources import generator
from datatable import dt

import numpy as np
import time

if __name__ == "__main__":

	# class to predict value
	recommender = Recommender()

	# Fetch initial data 

	initial = time.time()
	dataset = dt.fread("./resources/output/dataset.csv")
	print(str(round(time.time() - initial, 3)) + "s to read dataset")

	drows, dcols = dataset.shape

	if drows == 0:
		print("The dataset is empty!")
		exit(1)

	# features of dataset to parse queries
	recommender.datasetFeatures = list(dataset.names)[1::]

	initial = time.time()
	users = dt.fread("./resources/output/users.csv", header=False)
	print(str(round(time.time() - initial, 3)) + "s to read users")

	urows, ucols = users.shape

	if urows == 0:
		print("The user set is empty!")
		exit(1)

	initial = time.time()
	queries, queriesIDs = recommender.parse_queries("./resources/output/queries.csv")
	print(str(round(time.time() - initial, 3)) + "s to read queries")

	qrows, qcols = queries.shape

	if qrows == 0:
		print("The query set is empty!")
		exit(1)

	initial = time.time()
	cols = ["user"] + queriesIDs 
	ratings = dt.fread("./resources/output/utility_matrix.csv", columns=cols)
	print(str(round(time.time() - initial, 3)) + "s to read utility matrix")
	
	urows, ucols = ratings.shape

	if urows == 0:
		print("The utility matrix is empty!")
		exit(1)

	# Assign all values to recommender class
	recommender.init(users, queries, queriesIDs, dataset, ratings)

	# =========================== PART A ===========================

	to_predict, predictions, missed = recommender.compute_scores()

	if len(to_predict) > 0:
		print("\nFINAL PREDICTIONS [{} scores to predict, {} scores missed - {}% miss]:".format(len(to_predict), len(missed), round((len(missed) / len(to_predict)) * 100, 3)))	
	print(predictions)

	#print(predictions.max().to_numpy(), max(predictions.max().to_numpy()))
	exit(0)

	# Save prediction in csv file using generator csv writer
	
	command = ""
	while not command.lower() in ["yes", "no"]:
		command = input("Do you want to export final predictions? [Yes-No][Default: Yes] ")
		if command == "":
			command = "yes"

	if command == "yes":
		csv_rows = predictions.to_numpy().tolist()
		
		for ind in range(len(list(predictions.index.values))):
			csv_rows[ind].insert(0, predictions.index.values[ind])

		generator.write_csv("final_predictions", predictions.columns, csv_rows)

		print("Final utility matrix saved in output/final_predictions.csv")

	recommender.top_k_queries(to_predict, predictions, missed)

	exit(0)


