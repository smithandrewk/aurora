import tensorflow as tf
from keras.models import load_model
import numpy as np
from pandas import read_csv,DataFrame

def preprocess_file(dir, file):
    from pandas import read_excel
    from pandas import DataFrame, concat
    print(f'preprocessing {dir}/{file}')

    df = read_excel(f'{dir}/{file}')
    print(f'Length before:{df.shape[0]}')
    df = df.drop([0])  # remove first row with units
    # df = df.dropna(axis=0)
    from sklearn.impute import SimpleImputer
    imp_mean = SimpleImputer()
    imp_mean.fit(df)
    df = DataFrame(imp_mean.transform(df))
    print(df[df.isna().any(axis=1)])
    labels = DataFrame()
    # remove time stamp column if it exists
    if(df.columns[0]=='Time Stamp'):
        df = df.drop([df.columns[0]], axis=1)
        print(f'{filename} Removed TimeStamp')


    new_column_names = ['0.0-0.5', '0.5-1.0', '1.0-1.5', '15-20',
                        '2.0-0.5', '2.5-3.0', '3.0-3.5', '3.5-4.0',
                        '4.0-4.5', '4.5-5.0', '5.0-5.5', '5.5-6.0',
                        '6.0-0.5', '6.5-7.0', '7.0-7.5', '7.5-8.0',
                        '8.0-0.5', '8.5-9.0', '9.0-9.5', '9.5-10.0',
                        '10.0-10.5', '10.5-11.0', '11.0-11.5', '11.5-12.0',
                        '12.0-12.5', '12.5-13.0', '13.0-13.5', '13.5-14.0',
                        '14.0-14.5', '14.5-15.0', '15.0-15.5', '15.5-16.0',
                        '16.0-16.5', '16.5-17.0', '17.0-17.5', '17.5-18.0',
                        '18.0-18.5', '18.5-19.0', '19.0-19.5', '19.5-20.0',
                        'EMG', 'Activity']

    df = df.rename(columns={key: val for key,
                   val in zip(df.columns, new_column_names)})
    for col in df.loc[:, :]:  # typecast each column to type float
        df[col] = df[col].astype(float)
    filename = file.replace(file.split('.')[1], 'csv')
    df = concat([labels, df], axis=1)

    from os import system
    system(f'mkdir data/2_preprocessed')
    df.to_csv(f'data/2_preprocessed/{filename}', index=False)
    print(f'Length After:{df.shape[0]}')

def window_and_score_data(dir, file):
    print(f'{dir}/{file}')
    scaled = read_csv(f'{dir}/{file}')
    print(f'Length before:{scaled.shape[0]}')
    
    WINDOW_SIZE = 9
    labels = np.zeros(len(scaled))
    gen = tf.keras.preprocessing.sequence.TimeseriesGenerator(scaled, labels,length=WINDOW_SIZE, sampling_rate=1,batch_size=1,shuffle=False)
    model = load_model('model/mice_lstm_WIN-9.h5')
    pred = model.predict(gen).argmax(axis=1)
    pred_expert = pred.copy()
    expert=True
    if(expert):
        for j in range(len(pred_expert)-2):
            if((pred_expert[j:j+2]==np.array([2,0])).all()):
                pred_expert[j+1] = 2
    final = np.concatenate([[-1,-1,-1,-1],pred_expert,[-1,-1,-1,-1,-1]])
    final = DataFrame(final)
    final[final==-1] = 'X'
    final[final==0] = 'P'
    final[final==1] = 'S'
    final[final==2] = 'W'
    final = DataFrame(final)
    final.to_csv(f'data/4_scored/{file}',index=False)
    print(f'Length After:{final.shape[0]}')

def score_data_ann(dir, target_filename, model_filename):
    import pandas as pd
    from pandas import read_csv
    import numpy as np
    import os

    filename = f'{dir}/{target_filename}'
    X = read_csv(filename)
    X = np.array(X)

    from keras.models import load_model
    model = load_model('model/'+model_filename)
    x = np.array(X)
    y = model.predict(x)
    y = np.array(y)
    y = np.argmax(y,axis=1)
    if ( not os.path.isdir('data/predictions_ann')):
        os.system('mkdir data/predictions_ann')
    target_filename = target_filename.replace("_windowed_scaled.csv", "")
    # filename = "data/predictions/"+target_filename+"_scored_ann.csv"
    filename = "data/predictions_ann/"+target_filename+".csv"
    pd.DataFrame(y).to_csv(filename, index=False)
    print(filename)
def score_data_rf(dir, target_filename, model_filename):
    import joblib
    from pandas import read_csv
    import warnings
    warnings.filterwarnings("ignore")
    import pandas as pd
    import os

    filename = f'{dir}/{target_filename}'
    rf_model = joblib.load("model/"+model_filename)
    X = read_csv(filename)
    y = rf_model.predict(X)
    dct = {0: 0, 1: 0, 2: 0}
    for i in y:
        dct[i] += 1
    print(dct)
    if ( not os.path.isdir('data/predictions_rf')):
        os.system('mkdir data/predictions_rf')
    target_filename = target_filename.replace("_preprocessed_windowed.csv", "")
    # filename = "data/predictions/"+target_filename+"_scored_rf.csv"
    filename = "data/predictions_rf/"+target_filename+".csv"
    pd.DataFrame(y).to_csv(filename, index=False)
    print(filename)
def expand_predictions_ann(dir, file):
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    import os

    filename = f'{dir}/{file}'
    df = pd.read_csv(filename)
    Y = np.array(df)
    Y = Y.reshape(Y.shape[0],)
    lo_limit = len(Y)-1
    hi_limit = len(Y)+3
    Y_new = []
    for i,x in tqdm(enumerate(range(len(Y)+4))):
        if(i==0):
            Y_new.append(Y[0])
        elif(i<5):
            Y_new.append(np.argmax(np.bincount(Y[0:i])))
        elif(i>lo_limit and i!=hi_limit):
            Y_new.append(np.argmax(np.bincount(Y[lo_limit-(4-(i-lo_limit)):lo_limit])))
        elif(i==hi_limit): 
            Y_new.append(Y[lo_limit])
        else:
            Y_new.append(np.argmax(np.bincount(Y[i-4:i])))

    if ( not os.path.isdir('data/expanded_predictions_ann')):
        os.system('mkdir data/expanded_predictions_ann')
    
        print(Y.shape[0], len(Y_new))
    pd.DataFrame(Y_new).to_csv("data/expanded_predictions_ann/"+file,index=False)
def expand_predictions_rf(dir, file):
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    import os

    filename = f'{dir}/{file}'
    df = pd.read_csv(filename)
    Y = np.array(df)
    Y = Y.reshape(Y.shape[0],)
    lo_limit = len(Y)-1
    hi_limit = len(Y)+3
    Y_new = []
    for i,x in tqdm(enumerate(range(len(Y)+4))):
        if(i==0):
            Y_new.append(Y[0])
        elif(i<5):
            Y_new.append(np.argmax(np.bincount(Y[0:i])))
        elif(i>lo_limit and i!=hi_limit):
            Y_new.append(np.argmax(np.bincount(Y[lo_limit-(4-(i-lo_limit)):lo_limit])))
        elif(i==hi_limit):
            Y_new.append(Y[lo_limit])
        else:
            Y_new.append(np.argmax(np.bincount(Y[i-4:i])))

    if ( not os.path.isdir('data/expanded_predictions_rf')):
        os.system('mkdir data/expanded_predictions_rf')
    pd.DataFrame(Y_new).to_csv("data/expanded_predictions_rf/"+file,index=False)

def preprocess_zdb(dir, file):
    import sqlite3
    from sqlite3 import Error
    import os
    offset = 10e8
    recording_start_offset = -5 * 10e6

    filename = f'{dir}/{file}'
    new_dir = 'data/preprocessedZDB'
    if not os.path.isdir(new_dir):
        os.system(f'mkdir {new_dir}')
    os.system(f"cp '{filename}' {new_dir}/'{file}'")
    filename = f"{new_dir}/{file}"

    try:
        conn = sqlite3.connect(filename)
    except Error as e:
        print(e)

    cur = conn.cursor()

    query = "SELECT name FROM sqlite_master WHERE type='table' AND name='scoring_marker';"
    cur.execute(query)
    name = cur.fetchall()
    if name:
        #if the scoring marker table exists then there is already a scoring and this step is not necessary
        print(file+" skipped: Already Scored")
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

def conversion_zdb(dir_csv, dir_zdb, csv, zdb, mode):
    import pandas as pd
    import sqlite3
    from sqlite3 import Error
    import os
    offset = 10e7       #epoch time period

    new_dir = f'data/ZDB_{mode}'
    if not os.path.isdir(new_dir):
        os.system(f'mkdir {new_dir}')
    
    csv_filename = f'{dir_csv}/{csv}'
    zdb_filename = f'{dir_zdb}/{zdb}'
    os.system(f"cp '{zdb_filename}' '{new_dir}/{zdb}'")
    filename = f'{new_dir}/{zdb}'

    df = pd.read_csv(csv_filename)
    try:
        conn = sqlite3.connect(filename)
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

def convert_zdb_lstm(dir_csv, dir_zdb, csv, zdb):
    """
    zdb conversion submodule for lstm pipeline
    """    
    import pandas as pd
    import sqlite3
    from sqlite3 import Error
    import os
    offset = 10e7       #epoch time period
    rename_dict = {'W':'Sleep-Wake', 'S':'Sleep-SWS', 'P':'Sleep-Paradoxical', 'X':''}

    csv_filename = f'{dir_csv}/{csv}'
    zdb_filename = f'{dir_zdb}/{zdb}'

    df = pd.read_csv(csv_filename)
    try:
        conn = sqlite3.connect(zdb_filename)
    except Error as e:
        print(e)

    cur = conn.cursor()

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

        score = rename_dict[df.at[i,'0']]
        #insert epoch
        query = f"""
                INSERT INTO scoring_marker 
                (starts_at, ends_at, notes, type, location, is_deleted, key_id)
                VALUES 
                ({start_time}, {stop_time}, '', '{score}', '', 0, {keyid});
                """ 
        cur.execute(query)

    conn.commit()
    conn.close()
    return