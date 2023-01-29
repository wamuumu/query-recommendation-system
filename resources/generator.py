from math import radians
from sklearn.metrics import DistanceMetric
from datatable import dt, f

import pandas as pd
import numpy as np
import random
import string
import time
import csv
import os

#constants
MAX_DATA = 10000
MAX_QUERIES = 1000
MAX_USERS = 1000

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

# query
queries = []
user_queries = {}
usersIDs = []
queriesIDs = []

random.seed(time.time())

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

def create_users():
	
	print("Generating users...")

	global user_tastes, male_names, female_names, addresses, occupations, usersIDs

	addresses['lat'] = np.radians(addresses['lat'])
	addresses['lng'] = np.radians(addresses['lng'])

	dist = DistanceMetric.get_metric('haversine')

	distances = dist.pairwise(addresses[['lat', 'lng']].to_numpy()) * 6373

	for i in range(MAX_USERS):	
		user = "U" + str(i+1)
		usersIDs.append(user)

		occupation_choice = random.randint(1, 5)
		age_choice = random.randint(1, 4)

		gender = random.randint(1, 100)
		gender = "F" if gender < FEMALE else "M" # if M then likes F or viceversa

		address_choice = random.randint(0, len(addresses) - 1)

		user_tastes[user] = {}
		user_tastes[user]["gender"] = gender
		user_tastes[user]["likes"] = "F" if gender == "M" else "M"

		top_k = np.argsort(distances[address_choice])
		top_k = top_k[distances[address_choice][top_k] <= 100]

		user_tastes[user]["location"] = addresses.loc[address_choice]['city']
		user_tastes[user]["addresses"] = list(addresses.loc[top_k]['city'].to_numpy())

		age = random.randint(MIN_ETA, MAX_ETA)

		if age - age_choice < MIN_ETA:
			user_tastes[user]["ages"] = list(range(MIN_ETA, age + age_choice))
		elif age + age_choice > MAX_ETA:
			user_tastes[user]["ages"] = list(range(age - age_choice, MAX_ETA))
		else:
			user_tastes[user]["ages"] = list(range(age - age_choice, age + age_choice))

		user_tastes[user]["ages"] = list(map(str, user_tastes[user]["ages"]))
		user_tastes[user]["occupations"] = random.sample(occupations, occupation_choice)


	with open("./output/users.csv", "w+") as file:
		for user in usersIDs:
			file.write("%s\n" % user)

	print("Users' set created and saved in /output/users.csv")

def create_queries():
	
	global male_names, female_names, user_tastes, addresses, occupations, queries, user_queries, queriesIDs, usersIDs

	print("Generating queries...")

	dataset = dt.fread("./output/dataset.csv")
	dataset[:] = dt.str64
	dataset = dataset.to_pandas()

	data = set()
	picked = set()

	initial = time.time()

	while len(data) < MAX_QUERIES:

		u = random.choice(list(user_tastes.keys()))
		ruser = user_tastes[u]
		
		user_queries[len(data)] = u

		queryID = "Q" + str(len(data) + 1)

		randomName, randomGender, randomAddress, randomAge, randomOccupation = None, None, None, None, None
		query = []

		choice = random.randint(0, 5) #try to pick name for query
		if choice == 5:
			if ruser["likes"] == "F":
				tmp = female_names[random.randint(0, len(female_names) - 1)]
			else:
				tmp = male_names[random.randint(0, len(male_names) - 1)]
			randomName = "name=" + tmp
			query.append('name=="' + tmp + '"')
		elif choice == 0:
			randomGender = "gender=" + ruser["likes"]
			query.append('gender=="' + ruser["likes"] + '"')

		choice = random.randint(0, 1) #try to pick address for query
		if choice == 1:
			tmp = ruser["addresses"][random.randint(0, len(ruser["addresses"]) - 1)]
			randomAddress = "address=" + tmp
			query.append('address=="' + tmp + '"')

		choice = random.randint(0, 1) #try to pick age for query
		if choice == 1:
			tmp = str(ruser["ages"][random.randint(0, len(ruser["ages"]) - 1)])
			randomAge = "age=" + tmp 
			query.append('age=="' + tmp + '"')

		choice = random.randint(0, 1) #try to pick occupation for query
		if choice == 1:
			tmp = ruser["occupations"][random.randint(0, len(ruser["occupations"]) - 1)]
			randomOccupation = "occupation=" + tmp
			query.append('occupation=="' + tmp + '"')

		item = (queryID, randomName, randomGender, randomAddress, randomAge, randomOccupation)

		if (randomName == None and randomGender == None and randomAddress == None and randomAge == None and randomOccupation == None):
			continue
		else:

			query = " and ".join(query)

			filteredDataset = dataset.query(query)

			if len(filteredDataset.index.values) > 0:
				
				found = False
				for q in data:
					if q[1::] == item[1::]:
						found = True

				if not found:
					queriesIDs.append(queryID)
					queries.append(query)
					data.add(item)

					if len(data) % max(1, round(MAX_QUERIES * 0.1)) == 0:
						print("{} / {} [{}s]".format(len(data), MAX_QUERIES, round(time.time() - initial, 3)))
			else:
				continue
			

	data = [tuple(attr for attr in item if attr is not None) for item in data] #remove from all queries those attributes with no value

	mapped_data = list(map(list, data))
	sorted_queries = sorted(mapped_data, key=lambda tup: int(tup[0][1::]))

	write_csv("queries", None, sorted_queries)

	print("Queries' set created and saved in /output/queries.csv")
	
def create_matrix():
	
	global user_tastes, queries, user_queries, usersIDs, queriesIDs

	dataset = dt.fread("./output/dataset.csv")
	dataset[:] = dt.str64
	dataset = dataset.to_pandas()

	scores = np.full((MAX_USERS, MAX_QUERIES), "", dtype=object)

	fweights = {"gender": 0.5, "address": 0.25, "age": 0.15, "occupation": 0.10}
	#fweights = {"gender": 0.25, "address": 0.25, "age": 0.25, "occupation": 0.25}

	print("Generating partial utility matrix...")

	for u in user_tastes:
		user_tastes[u]["addresses"] = set(user_tastes[u]["addresses"])
		user_tastes[u]["ages"] = set(user_tastes[u]["ages"])
		user_tastes[u]["occupations"] = set(user_tastes[u]["occupations"])

	#print(user_queries)
	initial = time.time()
	for q in range(len(queries)):

		filteredDataset = dataset.query(queries[q]).to_numpy().T

		# 1 -> name, 2 -> gender, 3 -> address, 4 -> age, 5 -> occupation

		qgender = list(filteredDataset[2])
		qaddress = list_to_dict(list(filteredDataset[3]))
		qages = list_to_dict(list(filteredDataset[4]))
		qoccupation = list_to_dict(list(filteredDataset[5]))

		total = len(qgender)

		u_key = 0
		for u in user_tastes:
			
			choice = random.randint(0, 3)

			if choice <= 0 or user_queries[q] == u:
				
				score = qgender.count(user_tastes[u]['likes']) * fweights["gender"]
				score += intersection_score(qaddress, user_tastes[u]["addresses"]) * fweights["address"]
				score += intersection_score(qages, user_tastes[u]["ages"]) * fweights["age"]
				score += intersection_score(qoccupation, user_tastes[u]["occupations"]) * fweights["occupation"] 
				
				score = max(1, round((score / total) * 100)) if total > 0 else 1

				scores[u_key][q] = score

			u_key += 1

		if q != 0 and q % max(1, round(len(queries) * 0.1)) == 0:
			print("{} / {} [{}s]".format(q, len(queries), round(time.time() - initial, 3)))


	absMax = 100 / np.percentile(scores[scores != ""], 99.9)
	print(absMax)

	values = scores[scores != ""] * absMax
	values = np.array([round(i) for i in values])
	values[values > 100] = 100
	scores[scores != ""] = values

	usersIDs = np.array([usersIDs])
	scores = np.concatenate((usersIDs.T, scores), axis=1)

	write_csv("utility_matrix", queriesIDs, scores)

	print("Partial utility matrix created and saved in /output/utility_matrix.csv")


def intersection_score(queryValues, tasteSet):
	score = 0
	for t in tasteSet:
		if t in queryValues:
			score += queryValues[t]

	return score

def list_to_dict(llist):
	counter = {}
	for i in llist:
		if i not in counter:
			counter[i] = 0
		counter[i] += 1

	return counter


if __name__ == "__main__":

	get_data()

	create_dataset()
	create_users()
	create_queries()
	create_matrix()

	exit(0)
