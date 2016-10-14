Repo for <a href="https://www.kaggle.com/c/whats-cooking/"> Kaggle Whats Cooking Competition</a>. The goal is to create a machine learning model that can predict the type of cuisine based on recipe ingredients.

Currently, my best submission accuracy is 0.78379 using Logistic Regression. The best in the competition was 0.83216

<b>preProcessing.py</b> - Contains functions for preprocessing data for use in a document term model made with CountVectorizer()

<b>modelComparison.py</b> - Runs a few different machine learning classifiers to compare accuracies. Checks the following classifiers: Multinomial Naive Bayes, Logistic Regression, Random Forest, Decision Tree, Extra Trees, and SVM. This script is to get a general idea of what classifier performs the best

<b>train.json</b> - File from Kaggle containing all data and labels for training

<b>test.json</b> - File from Kaggle containing only the data and no labels. Labels will be predicted for each row of data and submitted to Kaggle for a score

<b>sample_submission.csv</b> - File from Kaggle outlining sample submission format

<b>finalModel.py</b> - Trains on whole set of training data and predicts values for testing data. Creates a submission file for Kaggle as well

<b>featureTuning</b> - Used to experiment with different feature combinations

<b>submission1.csv</b> - 1st submission to competition
