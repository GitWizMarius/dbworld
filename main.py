# This is a Script for Importing the DBWorld Mails and process them.
# The main.py contains the Workflow Logic while the other .py Files contain the actual "processes"
# Imports
import pandas as pd
from tqdm import tqdm

# Import own .py files
import db
import preprocess


def main():
    # Settings
    # To Control what you want to do
    keyword = True
    classification = False
    # Write Results to Database, Save Local as csv or Both
    csv = True
    database = True
    if database:
        db.connect()

    # Import DataSet and preprocess
    # Todo: Rework Import_Data -> preprocess.py will be init.py in new Version
    dataset = preprocess.import_data()
    print('DataSet imported with following DataTypes:')
    print(dataset.dtypes)

    # Process for each Mail is exactly the same
    for i in tqdm(range(len(dataset))):
        # Todo: Complete Process (Keyword Extraction, Classification, ...)
        # For Each Step we use the Base DataSet after it was Imported
        # Data Cleansing is done separate for each step
        if keyword:
            print('Keyword Extraction')

        if classification:
            print('Mail Classification')

        # Todo: Write all to DataBase after element was processed
        # Todo: Implement all needed Functions
        if database:
            print('Write to Database')

        # Todo: Implement .csv Export for all (Maybe as JSON Format?)
        if csv:
            print('Write to .csv')

        print('Just Some')

    db.disconnect()
    print("Done")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

'''#TODO: Complete Rework of Main and other Parts
def main():
    # Import DataSet and preprocess
    dataset = preprocess.import_data()
    print(dataset.dtypes)

    # Process for each Mail is exactly the same
    # for i in range(len(dataset)):
    # print(DataSet.loc[i, "Date_Received"], DataSet.loc[i, "From_Name"])

    for i in tqdm(range(len(dataset))):
        # print('Mail %s from %d' % (i + 1, len(DataSet.index)))
        while True:
            mail_id = db.insert_mail(dataset.loc[i, "Date_Received"], dataset.loc[i, "From_Name"],
                                     dataset.loc[i, "From_Mail"],
                                     dataset.loc[i, "Subject"], dataset.loc[i, "Body"])
            mail_id -= 1
            if mail_id == 0:
                print('Mail ID is ZERO')
                continue
            else:
                break

        # print(DataSet.loc[i, "Date_Received"])
        # Insert a Database Check if the content is doubled -> Sometimes a User resents the mail
        # These Keywords should be flagged as doubled
        keyword_text = dataset.loc[i, "Subject"] + '. ' + dataset.loc[i, "Body"]
        single_keywords = preprocess.keywords_single(keyword_text)
        multiple_keywords = preprocess.keywords_multiple(keyword_text)

        db.insert_singlekeyword(single_keywords, mail_id)
        db.insert_multiplekeyword(multiple_keywords, mail_id)

    print("Done")'''
