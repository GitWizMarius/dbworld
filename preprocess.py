# Imports
# Graphical Import
from tkinter import Tk  # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename
import re
import pandas as pd
from keybert import KeyBERT


# Loading the Model outside of the Definiton instead of inside
# makes the Keyword Extraction faster because it only has to load once instead of in each iteration
kw_model = KeyBERT(model='all-mpnet-base-v2')


# Import the DataSet with correct encoding and Rename Columns
def import_data():
    # Select Filepath from DataSet
    Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    filepath = askopenfilename()  # show an "Open" dialog box and return the path to the selected file
    try:
        # File Import with UTF8 encoding throws an Error
        df = pd.read_csv(filepath, encoding='ISO-8859-1', parse_dates=True)
        df = df.rename(columns={'Date Received': 'Date_Received', 'Sent on behalf of (display)': 'From_Name',
                                'Sent on behalf of (address)': 'From_Mail'})
        df['Subject'] = df['Subject'].str[10:]
        # Changing Data Types
        df = df.convert_dtypes()
        df['Date_Received'] = pd.to_datetime(df['Date_Received'], dayfirst=True)
        df = df.sort_values(by='Date_Received', ascending=False)
        df = df.reset_index(drop=True)
        df = df.fillna("Ops, something seems to be wrong here.")
    except pd.io.common.EmptyDataError:
        print("File is Empty/Etc.")
    finally:
        print(".CSV Import was successfully")
    return df


def keywords_single(text):
    # Extraction with KeyBERT -> Benchmark: https://towardsdatascience.com/keyword-extraction-a-benchmark-of-7-algorithms-in-python-8a905326d93f
    # Not as fast as Rake but better matches/results
    # Another Article: https://towardsdatascience.com/extracting-keyphrases-from-text-rake-and-gensim-in-python-eefd0fad582f
    # Addition: Maybe try one Version with Rake and one with KeyBERT later to compare result.
    # If there are Multiple Mentions of the exact same Keyword it onyl returns one of them
    # Maybe Lemmatize before so you dont get both: "(economic', 0.307), ('economics', 0.302)"
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words='english', top_n=50)
    return keywords


def keywords_multiple(text):
    # Extraction with KeyBERT -> Benchmark: https://towardsdatascience.com/keyword-extraction-a-benchmark-of-7-algorithms-in-python-8a905326d93f
    # Not as fast as Rake but better matches/results
    # Another Article: https://towardsdatascience.com/extracting-keyphrases-from-text-rake-and-gensim-in-python-eefd0fad582f
    # Addition: Maybe try one Version with Rake and one with KeyBERT later to compare result.
    # If there are Multiple Mentions of the exact same Keyword it onyl returns one of them
    # Maybe Lemmatize before so you dont get both: "(economic', 0.307), ('economics', 0.302)"
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(2, 3), stop_words='english', top_n=50)
    return keywords
