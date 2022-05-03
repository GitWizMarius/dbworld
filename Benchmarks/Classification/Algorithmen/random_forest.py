import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

########################################################################################################################

########################################################################################################################
# RandomForestClassifier
########################################################################################################################
# Cross-Validation for Algorithmen Tuning
def load(values):
    # Load the Data we created in feature.py
    # 1. File Paths
    path_df = '../Pickles/{}/df.pickle'.format(values)
    path_features_train = '../Pickles/{}/features_train.pickle'.format(values)
    path_labels_train = '../Pickles/{}/labels_train.pickle'.format(values)
    path_features_test = '../Pickles/{}/features_test.pickle'.format(values)
    path_labels_test = '../Pickles/{}/labels_test.pickle'.format(values)
    # 2. Load Pickles form File Path
    global df
    global features_train
    global labels_train
    global features_test
    global labels_test
    with open(path_df, 'rb') as data:
        df = pickle.load(data)
    with open(path_features_train, 'rb') as data:
        features_train = pickle.load(data)
    with open(path_labels_train, 'rb') as data:
        labels_train = pickle.load(data)
    with open(path_features_test, 'rb') as data:
        features_test = pickle.load(data)
    with open(path_labels_test, 'rb') as data:
        labels_test = pickle.load(data)

# Randomized Search Cross Validation
def rscv(values):
    print('#######################################################')
    print('{} - Start -> Randomized Search Cross Validation'.format(values))
    # Parameters
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=5)]  # Number of trees in forest
    max_features = ['auto', 'sqrt']  # Number of features to consider at every split
    max_depth = [int(x) for x in np.linspace(10, 110, num=5)]  # Maximum number of levels in tree
    max_depth.append(None)  # Use None for unlimited depth
    min_samples_split = [2, 5, 10]  # Minimum number of samples required to split a node
    min_samples_leaf = [1, 2, 4]  # Minimum number of samples required at each leaf node
    bootstrap = [True, False]  # Method of selecting samples for training each tree

    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    # Use the random grid to search for best hyperparameters

    # First create the base model to tune
    rf = RandomForestClassifier(random_state=8)  # Random Forest Classifier
    # random_state Controls both the randomness of the bootstrapping of the samples used when building trees (if bootstrap=True) and the sampling of the features to consider when looking for the best split at each node (if max_features < n_features)

    # Create the random search model
    # n_jobs = -1 means use all available CPUs
    rf_random_search = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=200, cv=3, verbose=1,
                                          random_state=8, n_jobs=-1, scoring='accuracy')

    # Fit the random search model
    rf_random_search.fit(features_train, labels_train)
    # Print the best parameters
    print('Best Algorithmen Parameters from randomized Search:')
    print(rf_random_search.best_params_)
    # Print the best score
    print('Best Score from randomized Search (Mean Accuracy):')
    print(rf_random_search.best_score_)

    # Print the best estimator
    print('Best Estimator:')
    print(rf_random_search.best_estimator_)
    print('#######################################################')

if __name__ == '__main__':
    # Please call only one Method at once
    # after that take the results and optimze parameters based on results before
    # Then start second Validation Process based on new parameters and use the best model for Classification
    print("Start")
    # Select Subject, Body or Both
    data = 'Subject'
    # Load Data from Pickles
    load(data)
    # Random Search Cross Validation
    rscv(data)
    # XX
