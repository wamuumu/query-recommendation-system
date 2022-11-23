import pandas as pd
import pandasql as ps

class Recommender:

	def __init__(self, dataset):
		self.dataset = dataset

	def compute_query(self):
		df = self.dataset
		return ps.sqldf("select * from df where id = 1") 


if __name__ == "__main__":
		
	users = pd.read_csv("./resources/output/users.csv", names = ["id"], header = None)
	dataset = pd.read_csv("./resources/output/dataset.csv", names = ["id","name","address","age","occupation"], header = 0)
	ratings = pd.read_csv("./resources/output/utility_matrix.csv")

	queriesIDs = list(ratings.columns)
	usersIDs = list(ratings.index.values)

	'''print(users)
	print(dataset)
	print(ratings)

	print(queriesIDs)
	print(usersIDs)'''

	recommender = Recommender(dataset)
	result = recommender.compute_query() #lista campi in conjunction

	print(result)

	exit(0)