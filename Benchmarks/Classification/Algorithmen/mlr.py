import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


########################################################################################################################
# Multinomial Logistic Regression
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
    C = [float(x) for x in np.linspace(start = 0.1, stop = 1, num = 10)] # Inverse of Regularization Strength
    multi_class = ['multinomial'] # Type of Classifier
    solver = ['newton-cg', 'sag', 'saga', 'lbfgs']  # Solver for the optimization problem
    class_weight = ['balanced', None] # Class Weight
    penalty = ['l2'] # Penalty

    # Create random grid
    random_grid = {
        'C': C,
        'multi_class': multi_class,
        'solver': solver,
        'class_weight': class_weight,
        'penalty': penalty
    }
    # First create the base model to tune
    mlr = LogisticRegression(random_state=8)

    # Create the random search model
    mlr_random_search = RandomizedSearchCV(
        estimator=mlr,
        param_distributions=random_grid,
        n_iter=80,
        scoring='accuracy',
        cv=3,
        verbose=1,
        random_state=8,
        n_jobs=-1,
    )
    # Fit the random search model
    mlr_random_search.fit(features_train, labels_train)
    # Print the best parameters
    print('Best Algorithm Parameters from randomized Search:')
    print(mlr_random_search.best_params_)
    # Print the best score
    print('Best Score from randomized Search (Mean Accuracy):')
    print(mlr_random_search.best_score_)
    # Print the best estimator
    print('Best Estimator:')
    print(mlr_random_search.best_estimator_)
    print('#######################################################')

    # Test best Estimator from Random Search instead of Grid Search
    if True:
        # Last Step: Save the model
        global best_mlr
        best_mlr = mlr_random_search.best_estimator_
        with open('../Pickles/Models/best_mlr_randomsearch_{}.pickle'.format(values), 'wb') as output:
            pickle.dump(best_mlr, output)

# Grid Search Cross Validation
def gscv(values):
    print('#######################################################')
    print('{} - Start -> Grid Search Cross Validation'.format(values))
    # Parameters (change manually base on rscv)
    C = [float(x) for x in np.linspace(start=0.1, stop=1, num=10)] # Inverse of Regularization Strength
    multi_class = ['multinomial'] # Type of Classifier
    solver = ['newton-cg'] # Solver for the optimization problem
    class_weight = [None] # Class Weight
    penalty = ['l2'] # Penalty

    '''
    {'solver': 'newton-cg', 'penalty': 'l2', 'multi_class': 'multinomial', 'class_weight': None, 'C': 0.6}
    '''


    # Create parameter grid
    mlr_param_grid = {
        'C': C,
        'multi_class': multi_class,
        'solver': solver,
        'class_weight': class_weight,
        'penalty': penalty,
    }
    # First create the base model to tune
    mlr = LogisticRegression(random_state=8)

    # Create splits in CV to be able to fix a random_stat
    # GridSearchCV does not have that argument -> use a Testsize of 1/3 (.33)
    cv_sets = ShuffleSplit(n_splits=3, test_size=.33, random_state=8)

    # Instantiate the grid search model
    mlr_grid_search = GridSearchCV(
        estimator=mlr,
        param_grid=mlr_param_grid,
        scoring='accuracy',
        cv=cv_sets,
        verbose=1,
        n_jobs=-1,
    )

    # Fit Grid Search Model to the given Data from Feature Engineering
    mlr_grid_search.fit(features_train, labels_train)

    print("Best Parameters using Grid Search:")
    print(mlr_grid_search.best_params_)
    print('Best Score from randomized Search (Mean Accuracy):')
    print(mlr_grid_search.best_score_)
    print('Best Estimator:')
    print(mlr_grid_search.best_estimator_)
    print('#######################################################')

    # Last step: Save the model
    global best_mlr
    best_mlr = mlr_grid_search.best_estimator_
    with open('../Pickles/Models/best_mlr_{}.pickle'.format(values), 'wb') as output:
        pickle.dump(best_mlr, output)

def fit(values):
    print('{} - Fit it like a Boss'.format(values))
    # Fit Model to Training Data
    best_mlr.fit(features_train, labels_train)

    # Get Predictions
    mlr_pred = best_mlr.predict(features_test)

    print('Training Accuracy: ')
    print(accuracy_score(labels_train, best_mlr.predict(features_train)))

    print('Test Accuracy: ')
    print(accuracy_score(labels_test, mlr_pred))

    print('Classification Report')
    print(classification_report(labels_test, mlr_pred))

    # Optional -> Confusion Matrix (good for Studienarbeit)
    if True:
        aux_df = df[['Classification', 'classification_codes']].drop_duplicates().sort_values('classification_codes')
        conf_matrix = confusion_matrix(labels_test, mlr_pred)
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
        plt.savefig('../Other/ConfusionMatrix_mlr_{}.png'.format(values), dpi=1200)

    # Model Summary for Later Comparsion
    sum = {
        'Model': 'Multinomial Logistic Regression',
        'Training Set Accuracy': accuracy_score(labels_train, best_mlr.predict(features_train)),
        'Test Set Accuracy': accuracy_score(labels_test, mlr_pred)
    }
    sum_model_mlr = pd.DataFrame(sum, index=[0])
    with open('../Pickles/Models/sum_model_mlr_{}.pickle'.format(values), 'wb') as output:
        pickle.dump(sum_model_mlr, output)



if __name__ == '__main__':
    # Please call only one Method at once
    # after that take the results and optimze parameters based on results before
    # Then start second Validation Process based on new parameters and use the best model for Classification
    # Select Subject, Body or Both
    data = 'Both'
    run = 1 # 1 is for First Step and 2 for Second Step
    # Load Data from Pickles
    load(data)

    if run == 1:
        # Random Search Cross Validation
        rscv(data)
        fit(data)
    elif run == 2:
        # Grid Search Cross Validation
        gscv(data)
        fit(data)