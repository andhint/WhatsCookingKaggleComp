import pandas as pd

from preProcessing import combineAndRemoveSpaces

from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import svm

from sklearn import metrics

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

# ### BUILD MODEL ############################################
# # instantiate Multinomial Naive Bayes model
# nb = MultinomialNB(alpha=0.1)

# # train model with X_test_dtm
# nb.fit(X_train_dtm, y_train)

# # make predictions
# y_pred_class = nb.predict(X_test_dtm)

# # CALCULATE ACCURACY 
# score = metrics.accuracy_score(y_test, y_pred_class)

# print "Score for Multinomial Naive Bayes:" , score
# # best accuracy : 0.746781979083 alpha=0.1
# ############################################################

# ### BUILD ANOTHER MODEL ####################################
# # instantiate Logistic Regression model
# logreg = LogisticRegression()

# # train model with X_test_dtm
# logreg.fit(X_train_dtm, y_train)

# # make predictions
# y_pred_class = logreg.predict(X_test_dtm)

# # CALCULATE ACCURACY
# score = metrics.accuracy_score(y_test, y_pred_class)
# print "Score for Logistic Regression:" , score
# # best accuracy : 0.776749798874
# ############################################################


# ### BUILD ANOTHER MODEL ####################################
# # instantiate Random Forest model
# rf = RandomForestClassifier()

# # train model with X_test_dtm
# rf.fit(X_train_dtm, y_train)

# # make predictions
# y_pred_class = rf.predict(X_test_dtm)

# # CALCULATE ACCURACY
# score = metrics.accuracy_score(y_test, y_pred_class)
# print "Score for Random Forest:" , score
# # best accuracy : 0.711082059533 n_estimators = 100, max_depth=None
# ############################################################

# ### BUILD ANOTHER MODEL ####################################
# # instantiate Decision Tree model
# dt = DecisionTreeClassifier()

# # train model with X_test_dtm
# dt.fit(X_train_dtm, y_train)

# # make predictions
# y_pred_class = dt.predict(X_test_dtm)

# # CALCULATE ACCURACY
# score = metrics.accuracy_score(y_test, y_pred_class)
# print "Score for Decision Tree:" , score
# # best accuracy : 0.601166532582
# ############################################################

# ### BUILD ANOTHER MODEL ####################################
# # instantiate Extra Trees model
# etc = ExtraTreesClassifier()

# # train model with X_test_dtm
# etc.fit(X_train_dtm, y_train)

# # make predictions
# y_pred_class = etc.predict(X_test_dtm)

# # CALCULATE ACCURACY
# score = metrics.accuracy_score(y_test, y_pred_class)
# print "Score for Extra Trees:" , score
# # best accuracy : 0.680510860821
# ############################################################

### BUILD ANOTHER MODEL ####################################
# instantiate SVM model
svm = svm.SVC(gamma=1, C=3.1622776601683795, probability=True)

# train model with X_test_dtm
svm.fit(X_train_dtm, y_train)

# make predictions
y_pred_class = svm.predict(X_test_dtm)

# CALCULATE ACCURACY
score = metrics.accuracy_score(y_test, y_pred_class)
print "Score for SVM:" , score
# best accuracy : 0.754726468222 kernel='linear', C=1
############################################################
