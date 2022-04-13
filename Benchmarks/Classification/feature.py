import pickle
import pandas as pd
import re
import nltk
# Downloading the stop words list
nltk.download('stopwords')
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
import numpy as np
# lemminflect shows as unused but needs to be imported!
import lemminflect
########################################################################################################################
# Settings
lemmatize = True

########################################################################################################################
# Load Dataset
from Benchmarks.Classification import process
df = process.import_data()
print('DataSet imported with following DataTypes:')
print(df.dtypes)
print(df.head())
print('========================================================')
########################################################################################################################
# Data Cleansing and Preprocessing
df['Body'] = df['Body'].str.replace(r"http\S+", " ")
df['Body'] = df['Body_'].str.replace("[^A-Za-z]+", " ")
df['Body'] = df['Body'].str.lower()

# Lemmatization der Bodys mit Spacy und Lemminflect
if lemmatize:
    lem_model = spacy.load("en_core_web_trf")
    nrows = len(df)
    for i in range(nrows):
        doc = lem_model(df.loc[i, 'Body'])
        df.loc[i, 'Body'] = ' '.join([token._.lemma() for token in doc])

# Stopwords entfernen
stop_words = list(stopwords.word('english'))
for stop_word in stop_words:
    regex = r"\b" + stop_word + r"\b"
    df['Body'] = df['Body'].str.replace(regex, '')

# Anmerkung: Zukünftig sollen unter anderem Mails klassifiziert werden, welche nicht in dem Datensatz vorhanden sind
# Dementsprechend sollten beim Preprocessing nahezu alle möglichen Szenarien berücksichtigt werden




doc = lem_model(text)
        # token._.lemma() -> uses lemminflect (pip install lemminflect): better results (example: data will not/less be transformed to datum)
        lem_text = " ".join([token._.lemma() for token in doc])

