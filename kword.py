import pandas as pd
import re
from keybert import KeyBERT
import spacy
# lemminflect shows as unused but needs to be imported!
import lemminflect

# Loading the Model outside of the Definiton instead of inside
# makes the Keyword Extraction faster because it only has to load once instead of in each iteration
kw_model = KeyBERT(model='all-mpnet-base-v2')
# Load Spacy en Model (NLP)
# Probably Fixed using lemminflect as addition: Lemmatize Sometimes changes data to Datum....
lem_model = spacy.load("en_core_web_trf")


def preprocess(text: str, lemmatize: bool) -> str:
    # Data Cleansing
    # remove links from text
    text = re.sub(r"http\S+", "", text)
    # remove numbers and special characters from text
    text = re.sub("[^A-Za-z]+", " ", text)
    # make all words lowercase
    text = text.lower().strip()
    # Replace some generic DBWorld Words :)
    text = text.replace('dbworld', '')
    text = text.replace('sigmod', '')
    # Lemmatize before so you dont get both: "(economic', 0.307), ('economics', 0.302)"
    # Todo: Check Keyword Results after Lemmatize -> especially Multiple Keyword!
    # Todo: Maybe use Multiple_Keywords without Lemmatize
    if lemmatize:
        doc = lem_model(text)
        # token._.lemma() -> uses lemminflect (pip install lemminflect): better results (example: data will not/less be transformed to datum)
        lem_text = " ".join([token._.lemma() for token in doc])

    return lem_text, text


def extract_all(text: str, lemmatize: bool, top_n: int):
    # Information: preprocess does not need to remove StopWords -> KeyBert includes this function
    # Info: Currently MultipleKeywords standard without Lemmatize
    singletext, multipletext = preprocess(text, lemmatize)

    single = kw_model.extract_keywords(singletext, keyphrase_ngram_range=(1, 1), stop_words='english', top_n=top_n)
    multiple = kw_model.extract_keywords(multipletext, keyphrase_ngram_range=(2, 3), stop_words='english', top_n=top_n)

    return single, multiple
