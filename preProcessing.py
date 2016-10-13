# 1) remove stopwords (optional)
# 2) remove spaces from individual features
# 3) combine all features in each list into one string for use with CountVectorizer() 

def combineAndRemoveSpaces(row):
	return " ".join([item.replace(' ', '') for item in row])

