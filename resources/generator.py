from math import radians, cos, sin, asin, sqrt

from datatable import dt, f
import pandas as pd

import numpy as np
import operator
import random
import string
import time
import csv
import os

#constants
MAX_DATA = 10000 #1000000
MAX_QUERIES = 10000 #10000
MAX_USERS = 10000 #100000
MIN_ETA, MAX_ETA = 18, 55
MIN_VOTE, MAX_VOTE = 1, 100

MALE = 75
FEMALE = 25

# data
male_names = []
female_names = []
addresses = []
occupations = []

# user
user_tastes = {}

random.seed(time.time())

allowed_features = ["name","gender","address","age","occupation"]

def get_data():

	global male_names, female_names, addresses, occupations

	with open("./input/male_names.txt") as file:
		male_names = [line.strip() for line in file]

	with open("./input/female_names.txt") as file:
		female_names = [line.strip() for line in file]

	addresses = dt.fread("./input/addresses.csv").to_pandas()

	with open("./input/occupations.txt") as file:
		occupations = [line.strip() for line in file]

	#print("names: " + str(len(names)))
	#print("addresses: " + str(len(addresses)))
	#print("occupations: " + str(len(occupations)))

def write_csv(filename, header, data):

	#header = [] -> array of attributes
	#data = [[],[],[],...] -> array of rows (attributes separated with commas)

	output_folder = os.path.exists("output")
	if not output_folder:
		os.makedirs("output")

	with open("./output/" + filename + ".csv", 'w+', newline='') as file:
		writer = csv.writer(file)

		#check if header is defined or not
		if header is not None:
			writer.writerow(header)

		#multiple rows writing
		writer.writerows(data)

def create_dataset():

	global male_names, female_names, addresses, occupations

	print("Generating dataset...")

	data = set()

	while len(data) < MAX_DATA:
		gender = random.randint(1, 100)
		gender = "F" if gender < FEMALE else "M" 

		if gender == "M":
			name = male_names[random.randint(0, len(male_names) - 1)]
		else:
			name = female_names[random.randint(0, len(female_names) - 1)]

		address_ind = random.randint(0, len(addresses) - 1)
		occupation_ind = random.randint(0, len(occupations) - 1)
		age = random.randint(MIN_ETA, MAX_ETA)

		item = (len(data) + 1, name, gender, addresses.iloc[address_ind]['city'], age, occupations[occupation_ind])

		if item in data and MAX_DATA < 1000:
			continue
		else:
			data.add(item)

	dataset = list(map(list, data))
	sorted_dataset = sorted(dataset, key=lambda tup: tup[0])

	write_csv("dataset", ["id", "name", "gender", "address", "age", "occupation"], sorted_dataset)

	print("Dataset created and saved in /output/dataset.csv")

def dist(lat1, long1, lat2, long2):

	lat1, long1, lat2, long2 = map(radians, [lat1, long1, lat2, long2])

	dlon = long2 - long1 
	dlat = lat2 - lat1 
	a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
	c = 2 * asin(sqrt(a)) 

	km = 6371 * c
	return km

def find_nearest(lat, lng, k):
	global addresses
	distances = addresses.apply(lambda row: dist(lat, lng, row['lat'], row['lng']), axis=1).to_dict()
	distances = list(dict(sorted(distances.items(), key=operator.itemgetter(1))).keys())[0:k]
	return ",".join(addresses.loc[distances, 'city'].to_numpy())

def create_users():
	
	print("Generating users...")

	global user_tastes, male_names, female_names, addresses, occupations

	data = []

	top_k_address = random.randint(0, 15)

	addresses['nearest'] = addresses.apply(lambda row: find_nearest(row['lat'], row['lng'], top_k_address), axis=1)

	print("nearest cities found")

	for i in range(MAX_USERS):
		user = "U" + str(i+1)
		data.append(user)

		occupation_choice = random.randint(1, 5)
		age_choice = random.randint(1, 4)

		gender = random.randint(1, 100)
		gender = "F" if gender > FEMALE else "M" # if M then likes F or viceversa

		age = random.randint(MIN_ETA, MAX_ETA)

		address_choice = random.randint(0, len(addresses) - 1)

		user_tastes[user] = {}
		user_tastes[user]["gender"] = gender
		user_tastes[user]["address"] = addresses.iloc[address_choice]['nearest'].split(",")

		if age == MIN_ETA:
			user_tastes[user]["age"] = list(range(age, age + age_choice))
		elif age == MAX_ETA:
			user_tastes[user]["age"] = list(range(age - age_choice, age))
		else:
			user_tastes[user]["age"] = list(range(age - age_choice, age + age_choice))

		user_tastes[user]["occupations"] = random.sample(occupations, occupation_choice)

	with open("./output/users.csv", "w+") as file:
		for user in data:
			file.write("%s\n" % user)

	print("Users' set created and saved in /output/users.csv")

def create_queries():
	
	global names, addresses, occupations

	print("Generating queries...")

	dataset = dt.fread("./output/dataset.csv")
	dataset[:] = dt.str32
	dataset = dataset.to_pandas()

	data = set()

	initial = time.time()

	while len(data) < MAX_QUERIES:

		queryID = "Q" + str(len(data) + 1)

		randomName, randomAddress, randomAge, randomOccupation = None, None, None, None #attributes that haven't been picked yet
		query = []

		choice = random.randint(0, 1) #try to pick name for query
		if choice == 1:
			tmp = names[random.randint(0, len(names) - 1)]
			randomName = "name=" + tmp
			query.append('name=="' + tmp + '"')

		choice = random.randint(0, 1) #try to pick address for query
		if choice == 1:
			tmp = addresses[random.randint(0, len(addresses) - 1)]
			randomAddress = "address=" + tmp
			query.append('address=="' + tmp + '"')

		choice = random.randint(0, 1) #try to pick age for query
		if choice == 1:
			tmp = str(random.randint(MIN_ETA, MAX_ETA))
			randomAge = "age=" + tmp 
			query.append('age=="' + tmp + '"')

		choice = random.randint(0, 1) #try to pick occupation for query
		if choice == 1:
			tmp = occupations[random.randint(0, len(occupations) - 1)]
			randomOccupation = "occupation=" + tmp
			query.append('occupation=="' + tmp + '"')

		item = (queryID, randomName, randomAddress, randomAge, randomOccupation)

		if (randomName == None and randomAddress == None and randomAge == None and randomOccupation == None):
			continue
		else:
			query = " and ".join(query)
			filteredDataset = dataset.query(query)

			if len(filteredDataset.index.values) > 0:
				data.add(item)
				print("{} / {} [{}s]".format(len(data), MAX_QUERIES, round(time.time() - initial, 3)))
			else:
				continue

	data = [tuple(attr for attr in item if attr is not None) for item in data] #remove from all queries those attributes with no value

	queries = list(map(list, data))
	sorted_queries = sorted(queries, key=lambda tup: int(tup[0][1::]))

	write_csv("queries", None, sorted_queries)

	print("Queries' set created and saved in /output/queries.csv")

def parse_queries(path:str):

	data = []
	indexes = []
	pdict = {}
	lineCount = 0

	with open(path) as f:
		for row in f:
			lineCount += 1
			row = row.rstrip('\n')
			values = row.split(",")
			indexes.append(values[0])
			values = values[1::]

			element = ["" for i in range(len(allowed_features))]

			for val in values:
				attr = val.split("=") #attr[0] -> feature's name, attr[1] -> feature's value
				ind = allowed_features.index(attr[0])
				element[ind] = attr[1]

			data.append(element)

	if lineCount > 0:
		data = np.array(data).transpose()

		for i in range(len(allowed_features)):
			pdict[allowed_features[i]] = data[i] 

	return dt.Frame(pdict), indexes
	
def create_matrix():
	
	print("Generating partial utility matrix...")

	users = dt.fread("./output/users.csv", header=False).to_numpy()
	queries, queriesIDs = parse_queries("./output/queries.csv")

	randomScores = np.random.randint(low=MIN_VOTE, high=MAX_VOTE, size=( len(users), len(queriesIDs))).astype('O')
	print("random scores generated")
	mask = np.random.randint(0, 5, size=randomScores.shape).astype(bool)
	print("mask generated")
	randomScores[np.logical_not(mask)] = ""
	print("mask applied")

	randomScores = np.concatenate((users, randomScores), axis=1)

	write_csv("utility_matrix", queriesIDs, randomScores)

	print("Partial utility matrix created and saved in /output/utility_matrix.csv")


if __name__ == "__main__":

	get_data()

	create_dataset()
	create_users()
	#create_queries()
	#create_matrix()

	exit(0)
