# This is a Script for Importing the DBWorld Mails and process them.
# In this case
# The main.py contains the Workflow Logic while the other .py Files containt the acutal processes

# Just to write a Log File from stdout
# Maybe add Timestamps later
class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)

    def flush(self):
        pass


import sys

f = open('logfile', 'w')
backup = sys.stdout
sys.stdout = Tee(sys.stdout, f)

# Imports
import pandas as pd
from tqdm import tqdm

# Import own .py files
import process
import keyword_algorithms

# Import DataSet and preprocess
DataSet, Base_Text = process.import_data()


def main():
    # n is how many mails should be used from the DataSet
    n = 3
    # Import DataSet and preprocess
    DataSet, Base_Text = process.import_data()

    print(DataSet.dtypes)
    print(Base_Text[:n])

    Text_Clean = Base_Text[:n].apply(lambda x: process.preprocess(x, r_stopwords=False))
    print(Text_Clean)

    results = keyword_algorithms.benchmark(Text_Clean[:n], shuffle=True)


if __name__ == "__main__":
    main()