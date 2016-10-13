Repo for <a href="https://www.kaggle.com/c/whats-cooking/"> Kaggle Whats Cooking Competition</a>. The goal is to create a machine learning model that can predict the type of cuisine based on recipe ingredients.

Currently, my best accuracy is 0.779867256637 using Logistic Regression

<b>preProcessing.py</b> - Contains functions for preprocessing data for use in a document term model made with CountVectorizer()

<b>modelComparison.py</b> - Runs a few different machine learning classifiers to compare accuracies. Checks the following classifiers: Multinomial Naive Bayes, Logistic Regression, Random Forest, Decision Tree, Extra Trees, and SVM. This script is to get a general idea of what classifier performs the best

<b>train.json</b> - File from Kaggle containing all data and labels for training

<b>test.json</b> - File from Kaggle containing only the data and no labels. Labels will be predicted for each row of data and submitted to Kaggle for a score

<b>sample_submission.csv</b> - File from Kaggle outlining sample submission format
