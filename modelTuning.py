import pandas as pd
import matplotlib.pyplot as plt
from preProcessing import combineAndRemoveSpaces

from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.grid_search import GridSearchCV

### READ IN DATA AND SPLIT INTO FEATURES (X) AND LABELS (y)
data = pd.read_json('train.json')
X = data.ingredients 
y = data.cuisine

#find null accuracy
nullAcc = max(pd.value_counts(y)) / float(len(y))
print 'Null accuracy:', nullAcc
############################################################

### PREPROCESS DATA
# remove spaces in individual strings then combine all features in list into one string
X = X.apply(combineAndRemoveSpaces)

############################################################


### SPLIT DATA INTO TESTING AND TRAINING SETS ##############
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)
############################################################

### VECTORIZING DATASET ####################################
# instantiate the vectorizer
vect = CountVectorizer()

# fit and transform into document-term matrix
X_train_dtm = vect.fit_transform(X_train)

# transform testing data (using fit from training data)
X_test_dtm = vect.transform(X_test)

############################################################

#### TEST DIFFERENT TUNING PARAMETERS ######################
c_range = [1,2,3,4,5,10,50,100]
c_scores = []
for c in c_range:
	# instantiate Logistic Regression model
	logreg = LogisticRegression(C=c)

	# train model with X_test_dtm
	logreg.fit(X_train_dtm, y_train)

	# make predictions
	y_pred_class = logreg.predict(X_test_dtm)

	# CALCULATE ACCURACY
	score = metrics.accuracy_score(y_test, y_pred_class)
	print score
	c_scores.append(score)
print c_scores

# plot values
plt.scatter(c_range, c_scores)
plt.xlabel('Value of C for Logistic Regression')
plt.ylabel('Testing Accuracy')
plt.show()