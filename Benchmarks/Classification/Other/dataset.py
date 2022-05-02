from Benchmarks.Classification import process
import pandas as pd
from autoplotter import run_app

df = process.import_data()

# Remove all rows containing specific string from a dataframe
df = df[~df['Classification'].str.contains('tbd')]
print(df.dtypes)


run_app(df)