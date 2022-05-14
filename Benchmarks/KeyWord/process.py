# Imports
# Graphical Import
from tkinter import Tk  # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename
import re

import nltk
import pandas as pd

from nltk.corpus import stopwords


# Import the DataSet with correct encoding and Rename Columns
def import_data():
    # Select Filepath from DataSet
    Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    filepath = askopenfilename()  # show an "Open" dialog box and return the path to the selected file
    try:
        # File Import with UTF8 encoding throws an Error
        df = pd.read_csv(filepath, encoding='ISO-8859-1', parse_dates=True)
        df = df.rename(columns={'Date Received': 'Date_Received',
                                'Date Sent': 'Date_Sent',
                                'Sent on behalf of (display)': 'From_Name',
                                'Sent on behalf of (address)': 'From_Mail',
                                'Body HTML': 'Body_HTML'})
        df['Subject'] = df['Subject'].str[10:]
        df['Subject'] = df['Subject'].str.replace("'", "")
        df['Subject'] = df['Subject'].str.replace('"', '')
        # Changing Data Types
        df = df.convert_dtypes()
        df['Date_Received'] = pd.to_datetime(df['Date_Received'], dayfirst=True)
        df['Date_Sent'] = pd.to_datetime(df['Date_Sent'], dayfirst=True)
        df = df.sort_values(by='Date_Received', ascending=False)
        df = df.reset_index(drop=True)
        df = df.fillna("Ops, something seems to be wrong here.")

        base_text = df["Subject"] + df["Body"]


    except pd.io.common.EmptyDataError:
        print("File is Empty/Etc.")
    finally:
        print(".CSV Import was successfully")
    return df, base_text


def preprocess(text: str, r_stopwords: bool) -> str:
    # remove links from text
    text = re.sub(r"http\S+", "", text)
    # remove numbers and special characters from text
    text = re.sub("[^A-Za-z]+", " ", text)
    # remove stopwords
    if r_stopwords:
        # 1. creates tokens
        tokens = nltk.word_tokenize(text)
        # 2. checks if token is a stopword and removes it
        tokens = [w for w in tokens if not w.lower() in stopwords.words("english")]
        # 3. joins all tokens again
        text = " ".join(tokens)
    # returns cleaned text
    text = text.lower().strip()
    return text
