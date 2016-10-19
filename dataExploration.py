import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from preProcessing import combineString
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_json('train.json')
X = data.ingredients 
y = data.cuisine

X = X.apply(combineString)
numbers = [str(num) for num in range(0,101)]
units = ['lbs', 'lb', 'g', 'oz', 'large', 'medium', 'small']
stopWords = [] + numbers + units

vect = CountVectorizer()
vect.fit(X)
X_dtm = vect.transform(X)
features = pd.DataFrame(X_dtm.toarray(), columns=vect.get_feature_names())

print 'Number of features: ', len(vect.get_feature_names())

counts = pd.Series(features.sum(axis=0), index=vect.get_feature_names())

counts = counts.sort_values(ascending=True)
print len(counts[counts > 20000])
