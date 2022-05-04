import pickle
from sklearn import svm
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


########################################################################################################################
# Support Vector Machine
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
    C = [.0001, .001, .01]  # Penalty Parameter
    gamma = [.0001, .001, .01, .1, 1, 10, 100] # Kernel Coefficient
    degree = [1, 2, 3, 4, 5] # Degree of the polynomial kernel
    kernel = ['linear', 'rbf', 'poly'] # Kernel Type
    probability = [True] # Whether to enable probability estimates

    # Create the parameter grid
    param_grid = {
        'C': C,
        'gamma': gamma,
        'kernel': kernel,
        'degree': degree,
        'probability': probability
    }
    # Base Model
    svc = svm.SVC(random_state=8)
    # Random Search Definition
    svm_random_search = RandomizedSearchCV(
        estimator=svc,
        param_distributions=param_grid,
        n_iter=200,
        scoring='accuracy',
        cv=3,
        verbose=1,
        random_state=8
    )
    # Fit to Base Model
    svm_random_search.fit(features_train, labels_train)

    # Print the best parameters
    print('Best Algorithm Parameters from randomized Search:')
    print(svm_random_search.best_params_)
    # Print the best score
    print('Best Score from randomized Search (Mean Accuracy):')
    print(svm_random_search.best_score_)
    # Print the best estimator
    print('Best Estimator:')
    print(svm_random_search.best_estimator_)
    print('#######################################################')


# Grid Search Cross Validation
def gscv(values):
    print('#######################################################')
    print('{} - Start -> Grid Search Cross Validation'.format(values))
    # Parameters -> using Parameters from Randomized Search (First Run)
    C = [.0001, .001, .01, .1] # Penalty Parameter
    degree = [3, 4, 5] # Degree of the polynomial kernel
    gamma = [1, 10, 100] # Kernel Coefficient
    probability = [True] # Whether to enable probability estimates

    # Create the parameter grid
    param_grid = [
        {'C': C, 'kernel': ['linear'], 'probability': probability},
        {'C': C, 'kernel': ['poly'], 'degree': degree, 'probability': probability},
        {'C': C, 'kernel': ['rbf'], 'gamma': gamma, 'probability': probability}
    ]

    # Base Model
    svc = svm.SVC(random_state=8)
    # Create splits in CV to be able to fix a random_stat
    # GridSearchCV does not have that argument -> use a Testsize of 1/3 (.33)
    cv_sets = ShuffleSplit(n_splits=3, test_size=.33, random_state=8)

    # Create a Grid Search Object
    svm_grid_search = GridSearchCV(
        estimator=svc,
        param_grid=param_grid,
        scoring='accuracy',
        cv=cv_sets,
        verbose=1
        )
    # Fit to the data
    svm_grid_search.fit(features_train, labels_train)

    # Print the best parameters
    print('Best Algorithm Parameters from Grid Search:')
    print(svm_grid_search.best_params_)
    # Print the best score
    print('Best Score from Grid Search (Mean Accuracy):')
    print(svm_grid_search.best_score_)
    # Print the best estimator
    print('Best Estimator:')
    print(svm_grid_search.best_estimator_)
    print('#######################################################')

    # Last Step: Save the model
    global best_svm
    best_svm = svm_grid_search.best_estimator_
    with open('../Pickles/Models/best_svm_{}.pickle'.format(values), 'wb') as output:
        pickle.dump(best_svm, output)

# Fit Model to our Traning Data and Test the "real" Performance :O
def fit(values):
    print('{} - Fit it like a Boss'.format(values))
    # Fit Model to Training Data
    best_svm.fit(features_train, labels_train)
    # Get Predictions
    svm_pred = best_svm.predict(features_test)

    print('Training Accuracy: ')
    print(accuracy_score(labels_train, best_svm.predict(features_train)))

    print('Testing Accuracy: ')
    print(accuracy_score(labels_test, svm_pred))

    print('Classification Report: ')
    print(classification_report(labels_test, svm_pred))

    # Optional -> Confusion Matrix (good for Studienarbeit)
    if True:
        aux_df = df[['Classification', 'classification_codes']].drop_duplicates().sort_values('classification_codes')
        conf_matrix = confusion_matrix(labels_test, svm_pred)
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
        plt.savefig('../Other/ConfusionMatrix_svm_{}.png'.format(values), dpi=1200)

    # Model Summary for Later Comparison
    sum = {
        'Model': 'Support Vector Machine',
        'Training Set Accuracy': accuracy_score(labels_train, best_svm.predict(features_train)),
        'Test Set Accuracy': accuracy_score(labels_test, svm_pred)
    }
    sum_model_svm = pd.DataFrame(sum, index=[0])
    with open('../Pickles/Models/sum_model_svm_{}.pickle'.format(values), 'wb') as output:
        pickle.dump(sum_model_svm, output)

if __name__ == '__main__':
    # Please call only one Method at once
    # after that take the results and optimze parameters based on results before
    # Then start second Validation Process based on new parameters and use the best model for Classification
    # Select Subject, Body or Both
    data = 'Subject'
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
