# Imports
# Graphical Import
from tkinter import Tk  # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename
import re
import pandas as pd


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
