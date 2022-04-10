import psycopg2
import os
from dotenv import load_dotenv

# Load .env files for Credentials
load_dotenv()
conn = None
cur = None


# Connect to DataBase
def connect():
    global conn
    global cur
    try:
        # Connect to an existing database
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(user=os.getenv('DB_USERNAME'),
                                password=os.getenv('DB_PASSWORD'),
                                host=os.getenv('DB_HOST'),
                                port=os.getenv('DB_PORT'),
                                database=os.getenv('DB_DATABASE'))
        # create a cursor
        cur = conn.cursor()

        # execute a statement and return current DB Version
        cur.execute('SELECT version()')
        db_version = cur.fetchone()  # returns tuple with 1 Element
        print('PostgreSQL database version: ' + db_version[0])
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            print('Database connection working like a charm.')


# Disconnect from DataBase
def disconnect():
    print(conn)
    conn.close
    print('Disconnected from the PostgreSQL database...')


# Todo: Redo the db Functions as needed
def old_insert_allmails(df):
    print("Write all to MailDatasetTable")
    query = "INSERT INTO maildatasettable(date_received, from_name, from_mail, subject, body) values(%s,%s,%s,%s,%s);"
    for i in range(len(df)):
        """ Execute a single INSERT request """
        try:
            cur.execute(query, (
                df.loc[i, "Date_Received"], df.loc[i, "From_Name"], df.loc[i, "From_Mail"], df.loc[i, "Subject"],
                df.loc[i, "Body"]))
            conn.commit()
        except (Exception, psycopg2.DatabaseError) as error:
            print("Error: %s" % error)
            conn.rollback()
            cur.close()
            return 1
        print(df.loc[i, "Subject"] + " was written to the DB")


def insert_mail(date_received, from_name, from_mail, subject, body, wordcount):
    """print("Insert Single Mail")"""
    query = "INSERT INTO maildatasettable (date_received, from_name, from_mail, subject, body, wordcount) values(%s,%s,%s,%s,%s,%s) RETURNING id;"

    """ Execute a single INSERT request and return ID """
    try:
        cur.execute(query, (date_received, from_name, from_mail, subject, body, wordcount))
        conn.commit()
        id_of_new_row = cur.fetchone()[0]

    except (Exception, psycopg2.DatabaseError) as error:
        print("Error: %s" % error)
        conn.rollback()
        cur.close()
        return 1
    return id_of_new_row + 1


def insert_singlekeyword(keywords, mail_id, doubled):
    """print("Insert Keywords to ID: ", mail_id)"""
    query = "INSERT INTO keywordssingle(id, keyword, doubled) values(%s, %s, %s)"

    for index, tuple in enumerate(keywords):
        """ Execute a single INSERT request """
        '''print('Insert Keyword:'+ tuple[0])'''
        try:
            cur.execute(query, (mail_id, tuple[0], doubled))
            conn.commit()
        except (Exception, psycopg2.DatabaseError) as error:
            print("Error: %s" % error)
            conn.rollback()
            cur.close()
            return 1


def insert_multiplekeyword(keywords, mail_id, doubled):
    """print("Insert Keywords to ID: ", mail_id)"""
    query = "INSERT INTO keywordsmultiple(id, keyword, doubled) values(%s, %s, %s)"
    for index, tuple in enumerate(keywords):
        """ Execute a single INSERT request """
        '''print('Insert Keyword:'+tuple[0])'''
        try:
            cur.execute(query, (mail_id, tuple[0], doubled))
            conn.commit()
        except (Exception, psycopg2.DatabaseError) as error:
            print("Error: %s" % error)
            conn.rollback()
            cur.close()
            return 1


def check_double(text):
    # Todo: Currently only simple check -> try complex one with PostgreSQL (currently not working idk why)
    query = "SELECT EXISTS(SELECT 1 FROM maildatasettable WHERE subject = {})".format(text)
    try:
        cur.execute(query)
        conn.commit()
        exists = cur.fetchone()[0]
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error: %s" % error)
        conn.rollback()
        cur.close()
        return 1

    return exists


def old_keybert_test(keywords, mail_id):
    # print("Insert Keyword:" + keywords + " to ID:" + mail_id)
    query = "INSERT INTO keywordssingle(id, keyword) values(%s, %s)"
    print(keywords)
    print(type(keywords))

    for index, tuple in enumerate(keywords):
        print(tuple[0])


def old_get_byname(name):
    print("Get Mails by Name:")
