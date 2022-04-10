# This is a Script for Importing the DBWorld Mails and process them.
# The main.py contains the Workflow Logic while the other .py Files contain the actual "processes"
# Imports
import pandas as pd
from tqdm import tqdm
import json
# Import own .py files
import db
import init
import kword


def main():
    # Keywords-Settings ====
    keywords = True  # Extract Keywords (True/False)
    top_n_keywords = 20  # Set how many Top Keywords to extract from each text
    lemmatize = True  # Lemmatize Single Keywords (True/False)

    # Classification-Settings ====
    classification = False  # Classification of Mails (True/False)

    # Export-Settings ====
    js = False  # Export as json (True/False)
    database = True  # Export/Write to Database (True/False)
    if database:
        db.connect()

    # Other Variables
    single_keywords = None
    multiple_keywords = None
    results = df = pd.DataFrame()

    # Import DataSet and preprocess
    dataset = init.import_data()
    print('DataSet imported with following DataTypes:')
    print(dataset.dtypes)
    print('========================================================')

    # Process for each Mail is exactly the same
    for i in tqdm(range(len(dataset))):
        # Todo: Complete Process (Keyword Extraction, Classification, ...)
        # For Each Step we use the Base DataSet after it was Imported
        # Data Cleansing is done separate for each step (Currently only single Keywords with Lemmatize)
        if keywords:
            keyword_text = dataset.loc[i, "Subject"] + '. ' + dataset.loc[i, "Body"]
            single, multiple, number_of_words = kword.extract_all(keyword_text, lemmatize, top_n_keywords)
            # print(single)
            # print(multiple)
            # print('========================================================')

        if classification:
            print('Mail Classification /tbd')

        # Todo: Write all to DataBase after element was processed
        # Todo: Implement all needed Functions
        if database:
            # Todo: Doubled Check -> Check if there is a similar mail (If yes -> mark Keywords as doubled)
            doubled = db.check_doubled(dataset.loc[i, "Subject"])
            # Write Mail to DB
            while True:
                mail_id = db.insert_mail(dataset.loc[i, "Date_Received"], dataset.loc[i, "From_Name"],
                                         dataset.loc[i, "From_Mail"],
                                         dataset.loc[i, "Subject"], dataset.loc[i, "Body"], number_of_words)
                mail_id -= 1
                if mail_id == 0:
                    print('Mail ID is ZERO')
                    continue
                else:
                    break
            # Write Keywords to DB
            if keywords:
                db.insert_singlekeyword(single, mail_id, doubled)
                db.insert_multiplekeyword(multiple, mail_id, doubled)
            # Write Classification to DB
            if classification:
                print('Write to Database')

        # Todo: Implement JSON Export containing all values
        # -> Late Project Stage if Time
        if js:
            # Save in .json Format or somethin else
            print('Write to .csv')

    if database:
        db.disconnect()
    if js:
        # Todo: Export df as json and format it nicely
        print('Export as json')
    print("Done")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
