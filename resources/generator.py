import argparse
import sys
import random
import csv
import string
import pandas as pd
import numpy as np
import time

#constants
MAX_DATA = 50000
MAX_QUERIES = 10000
MAX_USERS = 1000
MIN_ETA, MAX_ETA = 20, 30
MIN_VOTE, MAX_VOTE = 20, 100

#arrays of data
names = []
addresses = []
occupations = []

random.seed(time.time())

features = ["name","address","age","occupation"]

def get_data():

	global names, addresses, occupations

	with open("./old_input/names.txt") as file:
		names = [line.strip() for line in file]

	with open("./old_input/addresses.txt") as file:
		addresses = [line.strip() for line in file]

	with open("./old_input/occupations.txt") as file:
		occupations = [line.strip() for line in file]

	#print("names: " + str(len(names)))
	#print("addresses: " + str(len(addresses)))
	#print("occupations: " + str(len(occupations)))

def write_csv(filename, header, data):

	#header = [] -> array of attributes
	#data = [[],[],[],...] -> array of rows (attributes separated with commas)

	with open("./output/" + filename + ".csv", 'w+', newline='') as file:
		writer = csv.writer(file)

		#check if header is defined or not
		if header is not None:
			writer.writerow(header)

		#multiple rows writing
		writer.writerows(data)

def create_dataset():

	global names, addresses, occupations

	print("Generating dataset...")

	data = set()

	while len(data) < MAX_DATA:
		name_ind = random.randint(0, len(names) - 1)
		address_ind = random.randint(0, len(addresses) - 1)
		occupation_ind = random.randint(0, len(occupations) - 1)
		age = random.randint(MIN_ETA, MAX_ETA)

		item = (len(data) + 1, names[name_ind], addresses[address_ind], age, occupations[occupation_ind])

		'''if item in data:
			continue
		else:'''
		data.add(item)

	dataset = list(map(list, data))
	sorted_dataset = sorted(dataset, key=lambda tup: tup[0])

	write_csv("dataset", ["id", "name", "address", "age", "occupation"], sorted_dataset)

	print("Dataset created and saved in /output/dataset.csv")

def create_users():
	
	print("Generating users...")

	data = []

	for i in range(MAX_USERS):
		user = "U" + str(i+1)
		data.append(user)

	with open("./output/users.csv", "w+") as file:
		for user in data:
			file.write("%s\n" % user)

	print("Users' set created and saved in /output/users.csv")

def create_queries():
	
	global names, addresses, occupations

	print("Generating queries...")

	data = set()

	while len(data) < MAX_QUERIES:

		queryID = "Q" + str(len(data) + 1)

		randomID, randomName, randomAddress, randomAge, randomOccupation = None, None, None, None, None #attributes that haven't been picked yet

		'''choice = random.randint(0, 1) #try to pick id for query
		if choice == 1:
			randomID = "id=" + str(random.randint(1, MAX_DATA))'''

		choice = random.randint(0, 1) #try to pick name for query
		if choice == 1:
			randomName = "name=" + names[random.randint(0, len(names) - 1)]

		choice = random.randint(0, 1) #try to pick address for query
		if choice == 1:
			randomAddress = "address=" + addresses[random.randint(0, len(addresses) - 1)]

		choice = random.randint(0, 1) #try to pick age for query
		if choice == 1:
			randomAge = "age=" + str(random.randint(MIN_ETA, MAX_ETA))

		choice = random.randint(0, 1) #try to pick occupation for query
		if choice == 1:
			randomOccupation = "occupation=" + occupations[random.randint(0, len(occupations) - 1)]

		item = (queryID, randomID, randomName, randomAddress, randomAge, randomOccupation)

		if (randomID == None and randomName == None and randomAddress == None and randomAge == None and randomOccupation == None):
			continue
		else:
			data.add(item)

	data = [tuple(attr for attr in item if attr is not None) for item in data] #remove from all queries those attributes with no value

	queries = list(map(list, data))
	sorted_queries = sorted(queries, key=lambda tup: int(tup[0][1::]))

	write_csv("queries", None, sorted_queries)

	print("Queries' set created and saved in /output/queries.csv")

def parse_queries(path:str):

	data = []
	indexes = []
	pdict = {}

	with open(path) as f:
		for row in f:
			row = row.rstrip('\n')
			values = row.split(",")
			indexes.append(values[0])
			values = values[1::]

			element = ["" for i in range(len(features))]

			for val in values:
				attr = val.split("=") #attr[0] -> feature's name, attr[1] -> feature's value
				ind = features.index(attr[0])
				element[ind] = attr[1]

			data.append(element)

	data = np.array(data).transpose()

	for i in range(len(features)):
		pdict[features[i]] = data[i] 

	return pd.DataFrame(pdict, index = indexes)

def create_matrix():
	
	print("Generating partial utility matrix...")

	users = pd.read_csv("./output/users.csv", names = ["id"], header = None)['id'].to_numpy()
	queries = parse_queries("./output/queries.csv").index.values

	scores = []

	for u in users:
		user = []
		user.append(u)

		for i in range(len(queries)):

			choice = random.randint(0, 5) #try to evaluate query
			if choice <= 4:
				score = random.randint(MIN_VOTE, MAX_VOTE)
				user.append(score)
			else:
				user.append("")

		scores.append(user)

	write_csv("utility_matrix", queries, scores)

	print("Partial utility matrix created and saved in /output/utility_matrix.csv")


if __name__ == "__main__":

	#select the generation target
	parser = argparse.ArgumentParser(description="Data Generator")

	parser.add_argument('--func', nargs='?', default='')

	subparsers = parser.add_subparsers()
	
	parser_dataset = subparsers.add_parser('dataset', help='Create a new dataset')
	parser_dataset.set_defaults(func=create_dataset)

	parser_dataset = subparsers.add_parser('users', help='Create a new set of users')
	parser_dataset.set_defaults(func=create_users)

	parser_dataset = subparsers.add_parser('queries', help='Create a new set of queries')
	parser_dataset.set_defaults(func=create_queries)

	parser_dataset = subparsers.add_parser('matrix', help='Create a new partial utility matrix')
	parser_dataset.set_defaults(func=create_matrix)
	
	args = parser.parse_args()

	#try:
		#fetch all the data from files into arrays
	get_data()

	#execute the specified generative function
	#args.func()

	create_dataset()
	create_users()
	create_queries()
	create_matrix()

	#except:
		#print("Generative function not found! Please specify what needs to be generated...", file = sys.stderr)
		#exit(1)

	#exit(0)