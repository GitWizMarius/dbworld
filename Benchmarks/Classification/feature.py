import pickle
import pandas as pd
import re
import nltk
import openpyxl
# Downloading the stop words list
nltk.download('stopwords')
import spacy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
import numpy as np
# lemminflect shows as unused but needs to be imported!
import lemminflect

########################################################################################################################
# Settings
lemmatize = True
values = 'Subject'  # 'Subject' or 'Body' or 'Both' -> Select which Column to use for the feature extraction
representation = 1  # 1 = TF-IDF, 2 = XXX -> Todo: Implement other Text Representation Methods if Time is enough


def main():
    ####################################################################################################################
    # Load Dataset
    from Benchmarks.Classification import process

    df = process.import_data()
    print('DataSet imported with following DataTypes:')
    print(df.dtypes)
    print(df.head())
    print('========================================================')
    ####################################################################################################################
    # 1. Data Cleansing and Preprocessing
    df[values] = df[values].str.replace(r"http\S+", " ")
    df[values] = df[values].str.replace("[^A-Za-z]+", " ")
    df[values] = df[values].str.replace("'s", "")
    # Text to Lowercase so "Yolo" and "yolo" are the same
    df[values] = df[values].str.lower()

    # Lemmatization der Bodys mit Spacy und Lemminflect
    if lemmatize:
        lem_model = spacy.load("en_core_web_trf")
        nrows = len(df)
        for i in range(nrows):
            doc = lem_model(df.loc[i, values])
            df.loc[i, values] = ' '.join([token._.lemma() for token in doc])

    # Stopwords entfernen
    stop_words = list(stopwords.words('english'))
    for stop_word in stop_words:
        regex = r"\b" + stop_word + r"\b"
        df[values] = df[values].str.replace(regex, '')

    # Anmerkung: Zukünftig sollen unter anderem Mails klassifiziert werden, welche nicht in dem Datensatz vorhanden sind
    # Dementsprechend sollten beim Preprocessing nahezu alle möglichen Szenarien berücksichtigt werden
    ####################################################################################################################
    # 2. Label Coding
    # Create a Dictionary Containing the Different Categories/Groups and assign them a Number
    label_codes = {
        'CFP': 0,
        'Conference Announcement': 1,
        'Job Announcement': 2,
        'News': 3,
        'Workshop': 4,
    }
    # Create a new column in the DataFrame with copied Values from Classification Column
    df['classification_codes'] = df['Classification']
    # Change new Column to Values in the label_codes Dictionary
    df = df.replace({'classification_codes': label_codes})
    ####################################################################################################################
    # 3. Create Test Dataset to optimize the Modell Quality (Cross Validation to find the best Parameters)
    x_train, x_test, y_train, y_test = train_test_split(df[values],
                                                        df['classification_codes'],
                                                        test_size=0.2,
                                                        random_state=8)
    ####################################################################################################################
    # 4. Text Representation
    if representation == 1:
        # TF-IDF
        # Parameters
        ngram_range = (1, 2)  # (1,2) -> unigramm and bigramm, (1,3) -> unigramm, bigramm and trigramm
        min_df = 2  # Minimum Document Frequency
        max_df = 0.95  # Maximum Document Frequency
        max_features = 200  # None -> All Features, else -> Number of Features

        # Create TF-IDF Vectorizer
        tfidf_vectorizer = TfidfVectorizer(ngram_range=ngram_range,
                                           min_df=min_df,
                                           max_df=max_df,
                                           max_features=max_features,
                                           encoding='utf-8',
                                           lowercase=False,
                                           norm='l2',
                                           sublinear_tf=True)

        # Fit the Vectorizer to the Training Data
        features_train = tfidf_vectorizer.fit_transform(x_train).toarray()
        labels_train = y_train

        features_test = tfidf_vectorizer.transform(x_test).toarray()
        labels_test = y_test

        # Save TF-IDF Vectorizer
        with open('Pickles/tfidf.pickle', 'wb') as output:
            pickle.dump(tfidf_vectorizer, output)

    elif representation == 2:
        # Todo: Text / NLP based features -> If there is Time
        print('XXX')
    elif representation == 3:
        # Todo: Word Embeddings as features -> Word2Vec, ...
        print('XXX')
    else:
        print('Error: Representation not implemented')
    ####################################################################################################################
    # 5. Saving Feature
    with open('Pickles/x_train.pickle', 'wb') as output:
        pickle.dump(x_train, output)
    with open('Pickles/x_test.pickle', 'wb') as output:
        pickle.dump(x_test, output)
    with open('Pickles/y_train.pickle', 'wb') as output:
        pickle.dump(y_train, output)
    with open('Pickles/y_test.pickle', 'wb') as output:
        pickle.dump(y_test, output)
    with open('Pickles/features_train.pickle', 'wb') as output:
        pickle.dump(features_train, output)
    with open('Pickles/labels_train.pickle', 'wb') as output:
        pickle.dump(labels_train, output)
    with open('Pickles/features_test.pickle', 'wb') as output:
        pickle.dump(features_test, output)
    with open('Pickles/labels_test.pickle', 'wb') as output:
        pickle.dump(labels_test, output)
    with open('Pickles/df.pickle', 'wb') as output:
        pickle.dump(df, output)

    return df

if __name__ == '__main__':
    df = main()
    print('Done')
