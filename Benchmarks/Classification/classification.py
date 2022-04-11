###### Based on: https://github.com/miguelfzafra/Latest-News-Classifier/blob/master/0.%20Latest%20News%20Classifier/09.%20Report/Latest%20News%20Classifier.pdf

# Import DataSet and preprocess
from rich import columns

from Benchmarks.Classification import process

df = process.import_data()
print('DataSet imported with following DataTypes:')
print(df.dtypes)
print(df.head())
print('========================================================')

# Text Cleaning and df adjustment
df = df.drop(columns=['From_Name', 'From_Mail', 'Body_HTML'])




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