import pandas as pd
import sqlite3
from sqlite3 import Error

def preprocess_zdb(zdb):
    offset = 10e8
    recording_start_offset = -5 * 10e6
    try:
        conn = sqlite3.connect(zdb)
    except Error as e:
        print(e)

    cur = conn.cursor()

    query = "SELECT name FROM sqlite_master WHERE type='table' AND name='scoring_marker';"
    cur.execute(query)
    name = cur.fetchall()
    if name:
        #if the scoring marker table exists then there is already a scoring and this step is not necessary
        print(zdb+" skipped: Already Scored")
        return

    #get start time of recording to find when to start epochs
    query = """
            SELECT value
            FROM internal_property
            WHERE key = 'RecordingStart';
            """
    cur.execute(query)
    start_time = cur.fetchall()[0][0]
    start_time = int(start_time) + recording_start_offset
    stop_time = start_time + offset

    # create scoring marker table
    query = """
            CREATE TABLE IF NOT EXISTS scoring_marker (
                id INTEGER PRIMARY KEY, 
                starts_at INTEGER(8), 
                ends_at INTEGER(8), 
                notes TEXT, 
                type TEXT, 
                location TEXT, 
                is_deleted INTEGER(1), 
                key_id INTEGER
                );
            """
    cur.execute(query)

    #create scoring revision table
    query = """
            CREATE TABLE IF NOT EXISTS scoring_revision (
                id INTEGER PRIMARY KEY,
                name TEXT,
                is_deleted INTEGER(1),
                tags TEXT,
                version INTEGER(8),
                owner TEXT,
                date_created INTEGER(8)
                );
            """
    cur.execute(query)

    # insert needed rows into scoring_revision
    query = """
            INSERT INTO scoring_revision (name, is_deleted, tags, version, owner, date_created)
            VALUES ('Machine Data', 0, '', 0, '', 0);
            """
    cur.execute(query)
    query = """
            INSERT INTO scoring_revision (name, is_deleted, tags, version, owner, date_created)
            VALUES ('Machine Data', 0, '', 1, '', 0);
            """ 
    cur.execute(query)

    # insert needed rows into scoring_key
    query = """
            INSERT INTO scoring_key (date_created, name, owner, type)
            VALUES (0, '', '', 'Manual');
            """
    cur.execute(query)
    # insert needed rows into scoring_key
    query = """
            INSERT INTO scoring_key (date_created, name, owner, type)
            VALUES (0, '', '', 'Manual');
            """
    cur.execute(query)


    # create scoring_revision_to_key table
    query = """
            CREATE TABLE IF NOT EXISTS scoring_revision_to_key (
                revision_id INTEGER(8),
                key_id INTEGER(8)
                );
            """
    cur.execute(query)

    # insert necessary rows to scoring_revision_to_key
    query ="""
            INSERT INTO scoring_revision_to_key (revision_id, key_id)
            VALUES (1, 1);
            """
    cur.execute(query)

    query ="""
            INSERT INTO scoring_revision_to_key (revision_id, key_id)
            VALUES (2, 1);
            """
    cur.execute(query)

    query ="""
            INSERT INTO scoring_revision_to_key (revision_id, key_id)
            VALUES (2, 2);
            """
    cur.execute(query)

    # insert one score
    query = f"""
            INSERT INTO scoring_marker (starts_at, ends_at, notes, type, location, is_deleted, key_id)
            VALUES ({start_time}, {stop_time}, '', 'Sleep-Wake', '', 0, 2);
            """
    cur.execute(query)

    conn.commit()
    conn.close()

def ZDBConversion(csv, zdb):
    offset = 10e7       #epoch time period

    df = pd.read_csv(csv)
    try:
        conn = sqlite3.connect(zdb)
    except Error as e:
        print(e)

    #create sqlite table from csv    
    df.to_sql('raw_csv', conn, if_exists='replace', index=False)

    #copy csv data into formatted table    
    cur = conn.cursor()
    query = """
            CREATE TABLE IF NOT EXISTS temp_csv (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                status TEXT
            );
            """
    cur.execute(query)

    query = """
            INSERT INTO temp_csv (status)
            SELECT * FROM raw_csv;
            """
    cur.execute(query)

    #drop this table - creates issues
    query = "DROP TABLE IF EXISTS temporary_scoring_marker;"
    cur.execute(query)

    #get keyid of scoring
    query = "SELECT MAX(id) FROM scoring_revision WHERE name='Machine Data'"
    cur.execute(query)
    keyid = cur.fetchall()[0][0]

    #get starting point for scoring
    query = "SELECT id FROM scoring_marker WHERE type LIKE 'Sleep%' AND key_id='"+str(keyid)+"';"
    cur.execute(query)
    startid = cur.fetchall()[0][0]

    #get start time to crreate epochs
    query = 'SELECT starts_at FROM scoring_marker WHERE id = '+str(startid)+";"
    cur.execute(query)
    start_time = cur.fetchall()[0][0]
    stop_time = 0

    #delete first score before adding machine data
    query = "DELETE FROM scoring_marker;"
    cur.execute(query)


    #insert new epochs with scoring into the table
    for i in range(len(df)):
        #calculate epoch
        if i != 0:
            start_time = stop_time
        stop_time = start_time+offset

        #insert epoch
        query = f"""
                INSERT INTO scoring_marker 
                (starts_at, ends_at, notes, type, location, is_deleted, key_id)
                VALUES 
                ({start_time}, {stop_time}, '', '', '', 0, {keyid});
                """ 
        cur.execute(query)
        
        #get current id by selecting max id
        query = "SELECT MAX(id) from scoring_marker"
        cur.execute(query)
        currentid = cur.fetchall()[0][0]

        #set score
        query = f"""
                UPDATE scoring_marker
                SET type = (Select status
                            FROM temp_csv
                            WHERE id = {i+1})
                WHERE id = {currentid};
                """
        cur.execute(query)
    
    cur.execute("DROP TABLE temp_csv;")
    cur.execute("DROP TABLE raw_csv;")

    conn.commit()
    conn.close()
    return