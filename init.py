#
# This .py-File is used to Import the DataSet and preprocess it for general usage
#
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import pandas as pd


# Preprocess of the DataSet
# Contains Renaming, Data Converts, Sorts and handling NA Values
def preprocess(df):
    df = df.rename(columns={'Date Received': 'Date_Received', 'Sent on behalf of (display)': 'From_Name',
                            'Sent on behalf of (address)': 'From_Mail'})
    df['Subject'] = df['Subject'].str[10:]
    # Changing Data Types
    df = df.convert_dtypes()
    df['Date_Received'] = pd.to_datetime(df['Date_Received'], dayfirst=True)
    df = df.sort_values(by='Date_Received', ascending=False)
    df = df.reset_index(drop=True)
    df = df.fillna("Ops, something seems to be wrong here.")
    return df


def import_data():
    # Select .csv-File which you want to Import
    Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    filepath = askopenfilename()  # show an "Open" dialog box and return the path to the selected file
    try:
        # File Import with ISO-8859-1 encoding -> UTF-8 is throwing a Error
        df = pd.read_csv(filepath, encoding='ISO-8859-1', parse_dates=True)
    except pd.io.common.EmptyDataError:
        print("File is Empty/Etc.")
    finally:
        print(".CSV Import was successfully")
    return preprocess(df)
