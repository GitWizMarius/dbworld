import pickle
import pandas as pd
import nltk
from nltk.corpus import stopwords
nltk.download('wordnet')
nltk.download('omw-1.4')
import spacy
import lemminflect
lem_model = spacy.load("en_core_web_trf")

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Import already Trained Models and TF-IDF Object
# In this Case: Gradient Boosting Machine for Subject
gbm_path = 'Benchmarks/Classification/Pickles/Models/best_gbm_Subject.pickle'
with open(gbm_path, 'rb') as data:
    gbm_model = pickle.load(data)

tfidf_path = 'Benchmarks/Classification/Pickles/Subject/tfidf.pickle'
with open(tfidf_path, 'rb') as data:
    tfidf = pickle.load(data)

# Label Codes (same as in feature.py)
label_codes = {
    'CFP': 0,
    'Conference Announcement': 1,
    'Job Announcement': 2,
    'News': 3,
    'Workshop': 4,
}
stop_words = list(stopwords.words('english'))


def feature(text):
    # Dataframe creation
    lemmatized_text_list = []
    df = pd.DataFrame(columns=['text'])
    df.loc[0] = text
    # Data Cleansing
    df['text'] = df['text'].str.replace(r"http\S+", " ")
    df['text'] = df['text'].str.replace("[^A-Za-z]+", " ")
    df['text'] = df['text'].str.replace("'s", "")
    df['text'] = df['text'].str.lower()

    # Lemmatization
    doc = lem_model(df.iloc[0]['text'])
    df.loc[[0], 'text'] = ' '.join([token._.lemma() for token in doc])

    # Stop Words
    for stop_word in stop_words:
        regex_stopword = r"\b" + stop_word + r"\b"
        df['text'] = df['text'].str.replace(regex_stopword, '')
    df = df['text']

    # TF-IDF
    features = tfidf.transform(df).toarray()

    return features

def get_label_name(label):
    for key, value in label_codes.items():
        if value == label:
            return key


def predict(text):
    # Predict using GBM Model
    gbm_pred = gbm_model.predict(feature(text))[0]
    gbm_pred_proba = gbm_model.predict_proba(feature(text))[0]

    # Get Label Name
    label = get_label_name(gbm_pred)
    probability = gbm_pred_proba.max() * 100

    return label, probability

'''
if __name__ == '__main__':
    print(predict('Call for Paper - Last Call'))
'''
