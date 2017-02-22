from nltk.stem import PorterStemmer

# No longer using this fucntion, combineString yields better accuracy
def combineAndRemoveSpaces(row):
 	return " ".join([item.replace(' ', '') for item in row])

def combineString(row):
	return " ".join(row)


# no longer  in use, stemming decreases accuracy
def stemWords(row):
	ps = PorterStemmer()
	row = [ps.stem(word) for word in row.split(" ")]
	return row
