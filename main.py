import pandas as pd
from recommender import Recommender
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        elif isinstance(x, int):
        	return str(x)
        else:
            return ''

def create_text(x):
	return x['name'] + ' ' + x['age'] + ' ' + x['address'] + ' ' + x['occupation']

if __name__ == "__main__":
	
	features = ["name","address","age","occupation"]

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

	'''signatures = recommender.compute_signatures()
	similarity = recommender.compute_lsh(signatures)


	print()
	recommender.compute_query(queries['Q1'])
	print()
	recommender.compute_query(queries['Q2'])
	print()
	recommender.compute_query(queries['Q3'])'''

	features_dataset = dataset.drop(columns=['id'])
	for feature in features:
		features_dataset[feature] = features_dataset[feature].apply(clean_data)

	features_dataset['desc'] = features_dataset.apply(create_text, axis=1)

	print(features_dataset)

	# In CountVectorizer we only count the number of times a word appears in the document which results in biasing in favour of most frequent words.
	# In TfidfVectorizer we consider overall document weightage of a word. TfidfVectorizer weights the word counts by a measure of how often they appear in the documents.
	count = CountVectorizer(stop_words='english')
	count_matrix = count.fit_transform(features_dataset['desc'])

	# cosine similarity for different document size
	# jaccard for short documents of equal size
	print(count_matrix)
	cosine_sim = cosine_similarity(count_matrix, count_matrix)

	for r in range(cosine_sim.shape[0]):
		for c in range(cosine_sim.shape[0]):
			cosine_sim[r][c] = round(cosine_sim[r][c], 2)

	print()
	print(cosine_sim)
	print()

	for q in queriesIDs:
		recommender.compute_query(queries[q])
		print()


	exit(0)