import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


########################################################################################################################
# Multinomial Naive Bayes
# References:
# https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
# https://www.analyticsvidhya.com/blog/2020/11/understanding-naive-bayes-svm-and-its-implementation-on-spam-sms/
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

def mnb(values):
    load(values)
    # 1. Create the Model
    mnb = MultinomialNB()
    # 2. Train the Model
    mnb.fit(features_train, labels_train)
    # 3. Predict the Labels
    mnb_pred = mnb.predict(features_test)
    # 4. Evaluate the Model
    print('Training Accuracy: ')
    print(accuracy_score(labels_train, mnb.predict(features_train)))

    print('Test Accuracy: ')
    print(accuracy_score(labels_test, mnb_pred))

    print('Classification Report')
    print(classification_report(labels_test, mnb_pred))

    with open('../Pickles/Models/best_mnb_{}.pickle'.format(values), 'wb') as output:
        pickle.dump(mnb, output)

    # Optional -> Confusion Matrix (good for Studienarbeit)
    if True:
        aux_df = df[['Classification', 'classification_codes']].drop_duplicates().sort_values('classification_codes')
        conf_matrix = confusion_matrix(labels_test, mnb_pred)
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
        plt.savefig('../Other/ConfusionMatrix_mnb_{}.png'.format(values), dpi=1200)

    # Model Summary for later Comparison
    sum = {
        'Model': 'Multinomial Naïve Bayes',
        'Training Set Accuracy': accuracy_score(labels_train, mnb.predict(features_train)),
        'Test Set Accuracy': accuracy_score(labels_test, mnb_pred)
    }
    sum_model_mnb = pd.DataFrame(sum, index=[0])
    with open('../Pickles/Models/sum_model_mnb_{}.pickle'.format(values), 'wb') as output:
        pickle.dump(sum_model_mnb, output)

if __name__ == '__main__':
    # Select Subject, Body or Both
    data = 'Both'
    # Load Data from Pickles
    load(data)
    # Start mnb Model
    mnb(data)