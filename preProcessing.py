# 1) remove stopwords (optional)
# 2) remove spaces from individual features
# 3) combine all features in each list into one string for use with CountVectorizer() 

# No longer using this fucntion, combineString yields better accuracy
# def combineAndRemoveSpaces(row):
# 	return " ".join([item.replace(' ', '') for item in row])

def combineString(row):
	return " ".join(row)

# import pandas as pd


# data = pd.read_json('train.json')
# X = data.ingredients 
# print X[1]
# print combineString(X[1])
# stop = ['oil','eggs', 'milk']