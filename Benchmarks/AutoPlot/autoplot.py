from autoplotter import run_app
import pandas as pd

df = pd.read_csv('results.csv') # Reading data

run_app(df)
