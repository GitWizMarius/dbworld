import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


########################################################################################################################
# Gradient Boosting Machine
# References:
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
# https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/
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
    n_estimators = [200, 300, 400, 500, 600, 700, 800, 900, 1000]  # Number of Trees
    max_features = ['auto', 'sqrt']  # Number of Features
    max_depth = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # Max Depth
    max_depth.append(None)  # None
    min_samples_split = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # Min Samples Split
    min_samples_leaf = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Min Samples Leaf
    learning_rate = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # Learning Rate
    subsample = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # Subsample

    # Create the parameter grid
    param_grid = {
        'n_estimators': n_estimators,
        'max_features': max_features,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'learning_rate': learning_rate,
        'subsample': subsample
    }
    # Base Model
    gbm = GradientBoostingClassifier(random_state=8)

    # Random Search Definition
    gbm_random_search = RandomizedSearchCV(
        estimator=gbm,
        param_distributions=param_grid,
        n_iter=50,
        scoring='accuracy',
        cv=3,
        verbose=1,
        random_state=8,
        n_jobs=-1
    )
    # Fit to Base Model
    gbm_random_search.fit(features_train, labels_train)

    # Print the best parameters
    print('Best Algorithm Parameters from randomized Search:')
    print(gbm_random_search.best_params_)
    # Print the best score
    print('Best Score from randomized Search (Mean Accuracy):')
    print(gbm_random_search.best_score_)
    # Print the best estimator
    print('Best Estimator:')
    print(gbm_random_search.best_estimator_)
    print('#######################################################')

    # Test best Estimator from Random Search instead of Grid Search
    if True:
        # Last Step: Save the model
        global best_gbm
        best_gbm = gbm_random_search.best_estimator_
        with open('../Pickles/Models/best_gbm_randomsearch_{}.pickle'.format(values), 'wb') as output:
            pickle.dump(best_gbm, output)


# Grid Search Cross Validation
def gscv(values):
    print('#######################################################')
    print('{} - Start -> Grid Search Cross Validation'.format(values))
    # Parameters -> using Parameters from Randomized Search (First Run)
    max_depth = [35, 40, 45]  # Max Depth
    max_features = ['sqrt']  # Number of Features
    min_samples_leaf = [9]  # Min Samples Leaf
    min_samples_split = [80, 120]  # Min Samples Split
    n_estimators = [400]  # Number of Trees
    learning_rate = [.1, 1.]  # Learning Rate
    subsample = [.9]  # Subsample

    # Create the parameter grid
    param_grid = {
        'max_depth': max_depth,
        'max_features': max_features,
        'min_samples_leaf': min_samples_leaf,
        'min_samples_split': min_samples_split,
        'n_estimators': n_estimators,
        'learning_rate': learning_rate,
        'subsample': subsample
    }

    # Base Model
    gbm = GradientBoostingClassifier(random_state=8)

    # Create splits in CV to be able to fix a random_stat
    # GridSearchCV does not have that argument -> use a Testsize of 1/3 (.33)
    cv_sets = ShuffleSplit(n_splits=3, test_size=.33, random_state=8)

    # Create a Grid Search Object
    gbm_grid_search = GridSearchCV(
        estimator=gbm,
        param_grid=param_grid,
        scoring='accuracy',
        cv=cv_sets,
        verbose=1
    )

    # Fit to the data
    gbm_grid_search.fit(features_train, labels_train)

    # Print the best parameters
    print('Best Algorithm Parameters from Grid Search:')
    print(gbm_grid_search.best_params_)
    # Print the best score
    print('Best Score from Grid Search (Mean Accuracy):')
    print(gbm_grid_search.best_score_)
    # Print the best estimator
    print('Best Estimator:')
    print(gbm_grid_search.best_estimator_)
    print('#######################################################')

    # Last Step: Save the model
    global best_gbm
    best_gbm = gbm_grid_search.best_estimator_
    with open('../Pickles/Models/best_gbm_{}.pickle'.format(values), 'wb') as output:
        pickle.dump(best_gbm, output)


# Fit Model to our Traning Data and Test the "real" Performance :O
def fit(values):
    print('{} - Fit it like a Boss'.format(values))
    # Fit Model to Training Data
    best_gbm.fit(features_train, labels_train)
    # Predict on the Test Data
    gbm_pred = best_gbm.predict(features_test)

    print('Training Accuracy: ')
    print(accuracy_score(labels_train, best_gbm.predict(features_train)))

    print('Testing Accuracy: ')
    print(accuracy_score(labels_test, gbm_pred))

    print('Classification Report: ')
    print(classification_report(labels_test, gbm_pred))

    # Optional -> Confusion Matrix (good for Studienarbeit)
    if True:
        aux_df = df[['Classification', 'classification_codes']].drop_duplicates().sort_values('classification_codes')
        conf_matrix = confusion_matrix(labels_test, gbm_pred)
        # plt.figure(figsize=(12, 6)
        sns.heatmap(conf_matrix,
                    annot=True,
                    xticklabels=aux_df['Classification'].values,
                    yticklabels=aux_df['Classification'].values,
                    cmap="BuGn")
        plt.ylabel('Predicted')
        plt.xlabel('Actual')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        # Save Plot in 4k Resolution OmegaLuL
        plt.savefig('../Other/ConfusionMatrix_gbm_{}.png'.format(values), dpi=1200)

    # Model Summary for Later Comparison
    sum = {
        'Model': 'Gradient Boosting Machine',
        'Training Set Accuracy': accuracy_score(labels_train, best_gbm.predict(features_train)),
        'Test Set Accuracy': accuracy_score(labels_test, gbm_pred)
    }
    sum_model_gbm = pd.DataFrame(sum, index=[0])
    with open('../Pickles/Models/sum_model_gbm_{}.pickle'.format(values), 'wb') as output:
        pickle.dump(sum_model_gbm, output)


if __name__ == '__main__':
    # Please call only one Method at once
    # after that take the results and optimze parameters based on results before
    # Then start second Validation Process based on new parameters and use the best model for Classification
    # Select Subject, Body or Both
    data = 'Both'
    run = 2  # 1 is for First Step and 2 for Second Step
    # Load Data from Pickles
    load(data)

    if run == 1:
        # Random Search Cross Validation
        rscv(data)
        fit(data)  # Only run in First Step when Results from run 2 are worse than run 1
    elif run == 2:
        # Grid Search Cross Validation
        gscv(data)
        fit(data)
