import pandas as pd
import matplotlib.pyplot as plt
from preProcessing import combineAndRemoveSpaces
from collections import Counter

from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.grid_search import GridSearchCV

### READ IN DATA AND SPLIT INTO FEATURES (X) AND LABELS (y)
trainData = pd.read_json('train.json')
X_train = trainData.ingredients 
y_train = trainData.cuisine

testData = pd.read_json('test.json')
X_test = testData.ingredients
idNum = testData.id
############################################################

### PREPROCESS DATA ########################################
# remove spaces in individual strings then combine all features in list into one string
X_train = X_train.apply(combineAndRemoveSpaces)
X_test = X_test.apply(combineAndRemoveSpaces)
############################################################

### VECTORIZING DATASET ####################################
# instantiate the vectorizer
vect = CountVectorizer(max_features=6000)
	# default accuracy = 0.777152051488
	# 6264 features by default
	# best accuracry for max_features at 6000
	# min_features only decreases accuarcy
# fit and transform into document-term matrix
X_train_dtm = vect.fit_transform(X_train)

# transform testing data (using fit from training data)
X_test_dtm = vect.transform(X_test)
############################################################

#### TEST DIFFERENT TUNING PARAMETERS ######################
# instantiate Logistic Regression model
logreg = LogisticRegression(C=2)

# train model with X_test_dtm
logreg.fit(X_train_dtm, y_train)

# make predictions
y_pred_class = logreg.predict(X_test_dtm)

# concat
submission = pd.concat([idNum, pd.Series(y_pred_class)], axis=1)
submission.columns = ['id','cuisine']
submission.to_csv('submission1.csv', index=False)