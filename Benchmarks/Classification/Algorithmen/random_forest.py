import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


########################################################################################################################
# RandomForestClassifier
########################################################################################################################
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
    print('Best Algorithm Parameters from randomized Search:')
    print(rf_random_search.best_params_)
    # Print the best score
    print('Best Score from randomized Search (Mean Accuracy):')
    print(rf_random_search.best_score_)
    # Print the best estimator
    print('Best Estimator:')
    print(rf_random_search.best_estimator_)
    print('#######################################################')


# Grid Search Cross Validation
def gscv(values):
    print('#######################################################')
    print('{} - Start -> Grid Search Cross Validation'.format(values))
    # Parameters -> using Parameters from Randomized Search (First Run)
    # If max_depth is 60 than max_depth = [50, 60, 70] (same for min_samples_leaf and min_samples_split)
    bootstrap = [True]  # Method of selecting samples for training each tree
    max_depth = [50, 60, 70]  # Maximum number of levels in tree
    max_features = ['sqrt']  # Number of features to consider at every split
    min_samples_leaf = [1, 2, 4]  # Minimum number of samples required at each leaf node
    min_samples_split = [2, 5, 7]  # Minimum number of samples required to split a node
    n_estimators = [200]  # Number of trees in forest

    # Create the parameter grid
    param_grid = {
        'bootstrap': bootstrap,
        'max_depth': max_depth,
        'max_features': max_features,
        'min_samples_leaf': min_samples_leaf,
        'min_samples_split': min_samples_split,
        'n_estimators': n_estimators
    }

    # Base Model
    rf = RandomForestClassifier(random_state=8)

    # Create splits in CV to be able to fix a random_stat
    # GridSearchCV does not have that argument -> use a Testsize of 1/3 (.33)
    cv_sets = ShuffleSplit(n_splits=3, test_size=.33, random_state=8)

    # Grid Search Model
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=cv_sets,
        verbose=1,
        scoring='accuracy'
    )

    # Fit Grid Search Model to the given Data from Feature Engineering
    grid_search.fit(features_train, labels_train)

    print("Best Parameters using Grid Search:")
    print(grid_search.best_params_)
    print('Best Score from randomized Search (Mean Accuracy):')
    print(grid_search.best_score_)
    print('Best Estimator:')
    print(grid_search.best_estimator_)
    print('#######################################################')

    # Last Step: Save the model
    global best_rf
    best_rf = grid_search.best_estimator_
    with open('../Pickles/Models/best_rf_{}.pickle'.format(values), 'wb') as output:
        pickle.dump(best_rf, output)


# Fit Model to our Traning Data and Test the "real" Performance :O
def fit(values):
    print('{} - Fit it like a Boss'.format(values))
    # Fit Model to Training Data
    best_rf.fit(features_train, labels_train)
    # Get Predictions
    rf_pred = best_rf.predict(features_test)

    print('Training Accuracy: ')
    print(accuracy_score(labels_train, best_rf.predict(features_train)))

    print('Test Accuracy: ')
    print(accuracy_score(labels_test, rf_pred))

    print('Classification Report')
    print(classification_report(labels_test, rf_pred))

    # Optional -> Confusion Matrix (good for Studienarbeit)
    if True:
        aux_df = df[['Classification', 'classification_codes']].drop_duplicates().sort_values('classification_codes')
        conf_matrix = confusion_matrix(labels_test, rf_pred)
        #plt.figure(figsize=(12, 6)
        sns.heatmap(conf_matrix,
                    annot=True,
                    xticklabels=aux_df['Classification'].values,
                    yticklabels=aux_df['Classification'].values,
                    cmap="BuGn")
        plt.ylabel('Predicted')
        plt.xlabel('Actual')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        #Save Plot in 4k Resolution OmegaLuL
        plt.savefig('../Other/ConfusionMatrix_rf_{}.png'.format(values), dpi=1200)

    # Model Summary for Later Comparison
    sum = {
        'Model': 'Random Forest',
        'Training Set Accuracy': accuracy_score(labels_train, best_rf.predict(features_train)),
        'Test Set Accuracy': accuracy_score(labels_test, rf_pred)
    }
    sum_model_rf = pd.DataFrame(sum, index=[0])
    with open('../Pickles/Models/sum_model_svm_{}.pickle'.format(values), 'wb') as output:
        pickle.dump(sum_model_rf, output)


if __name__ == '__main__':
    # Please call only one Method at once
    # after that take the results and optimze parameters based on results before
    # Then start second Validation Process based on new parameters and use the best model for Classification
    # Select Subject, Body or Both
    data = 'Both'
    run = 2 # 1 is for First Step and 2 for Second Step
    # Load Data from Pickles
    load(data)

    if run == 1:
        # Random Search Cross Validation
        rscv(data)
    elif run == 2:
        # Grid Search Cross Validation
        gscv(data)
        fit(data)


    """
    ---- Auflistung der Werte aus Randomized Search Cross Validation ----
    --Subject:
    Best Algorithmen Parameters from randomized Search:
    {'n_estimators': 1550, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_features': 'auto', 'max_depth': 35, 'bootstrap': True} 
    Best Score from randomized Search (Mean Accuracy): 0.8395626748029494
    --Body:
    Best Algorithmen Parameters from randomized Search:
    {'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 60, 'bootstrap': False}
    Best Score from randomized Search (Mean Accuracy): 0.7407322654462242
    --Both:
    Best Algorithmen Parameters from randomized Search:
    {'n_estimators': 200, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 60, 'bootstrap': True}
    Best Score from randomized Search (Mean Accuracy): 0.7436816679379609
    """
