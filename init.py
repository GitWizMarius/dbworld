#
# This .py-File is used to Import the DataSet and preprocess it for general usage
#
import easygui
import re
import pandas as pd
from keybert import KeyBERT


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
    # Select Filepath from DataSet
    # Todo: Add Error Handling for Import
    filepath = easygui.fileopenbox()
    try:
        # File Import with ISO-8859-1 encoding -> UTF-8 is throwing a Error
        df = pd.read_csv(filepath, encoding='ISO-8859-1', parse_dates=True)
    except pd.io.common.EmptyDataError:
        print("File is Empty/Etc.")
    finally:
        print(".CSV Import was successfully")
    return preprocess(df)
