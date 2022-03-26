import psycopg2
import os
from dotenv import load_dotenv

# Load .env files for Credentials
load_dotenv()
conn = None


# Connect to DataBase
def connect():
    global conn
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
