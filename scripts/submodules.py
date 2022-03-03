def preprocess_excel(dir,filename):
    from os import system
    from pandas import read_excel,Categorical
    df = read_excel(f'{dir}/{filename}') # load xls file into pandas dataframe
    cols = df.columns # get column names
    new_cols = [] # initialize list to contain new column names

    # remove time stamp column if it exists
    if(cols[0]=='Time Stamp'):
        df = df.drop([df.columns[0]], axis=1)
        print(f'{filename} Removed TimeStamp')
    cols = df.columns
    
    if(cols[0]=="Rodent Sleep"):
        for i in range(len(cols)): # deep copy column names to new column names
            new_cols.append(cols[i])
        for i in range(1,len(new_cols)-2): # for each column name, try to extract frequency range
            new_cols[i]=new_cols[i].split(',')[0][7:]
        for i in range(1,6): # cleanup for first 5 frequency ranges still containing " HZ"
            new_cols[i]=new_cols[i][0:5]
        print(new_cols[1])
        if(new_cols[1]!="0-0.5"):
            new_cols[1]="0-0.5"
        new_cols[-1]=new_cols[-1][:8] # remove end of column name
        new_cols[-2]=new_cols[-2][:5] # remove end of column name
        df.columns=new_cols # set dataframe column names to new column names
        df.rename(columns={"Rodent Sleep":"Class"},inplace=True)
        df.drop(df[df['Class'] == "X"].index, inplace = True)
        df["Class"]=Categorical(df["Class"]).fillna(method='backfill').codes # Convert to categorical codes here so we can analyze percentage of each class in next code block
        df = df.drop([0]) # drop first row containing units [muV^2]
        df = df.fillna(0) ## handle NaN values
        for col in df.loc[:, df.columns != 'Class']: # typecast each column to type float
            df[col] = df[col].astype(float)
    elif(cols[0]=="10 second Epochs"):
        del df[df.columns[0]] # remove first column
        for i in range(1,len(cols)): # deep copy column names to new column names
            new_cols.append(cols[i])
        for i in range(2,len(new_cols)-2): # for each column name, try to extract frequency range
            new_cols[i]=new_cols[i].split(',')[0][7:]
        for i in range(2,7): # cleanup for first 5 frequency ranges still containing " HZ"
            new_cols[i]=new_cols[i][0:5]
        new_cols[-2]=new_cols[-2][:8] # remove end of column name
        new_cols[-1]=new_cols[-1][:5] # remove end of column name
        df.columns=new_cols # set dataframe column names to new column names
        df.rename(columns={"Rodent Sleep":"Class"},inplace=True)
        df.drop(df[df['Class'] == "X"].index, inplace = True)
        df["Class"]=Categorical(df["Class"]).codes # Convert to categorical codes here so we can analyze percentage of each class in next code block
        df = df.drop([0]) # drop first row containing units [muV^2]
        df = df.fillna(0) ## handle NaN values
        for col in df.loc[:, df.columns != 'Class']: # typecast each column to type float
            df[col] = df[col].astype(float)
    else:
        for i in range(len(cols)): # deep copy column names to new column names
            new_cols.append(cols[i])    
        for i in range(0,len(new_cols)-2): # for each column name, try to extract frequency range
            new_cols[i]=new_cols[i].split(',')[0][7:]
        for i in range(0,5): # cleanup for first 5 frequency ranges still containing " HZ"
            new_cols[i]=new_cols[i][0:5]
        new_cols[-1]=new_cols[-1][:8] # remove end of column name
        new_cols[-2]=new_cols[-2][:5] # remove end of column name
        df.columns=new_cols # set dataframe column names to new column names
        df = df.drop([0]) # drop first row containing units [muV^2]
        df = df.fillna(0) ## handle NaN values
        for col in df.loc[:, df.columns != 'Class']: # typecast each column to type float
            df[col] = df[col].astype(float)
    system('mkdir data/preprocessed')
    filename = filename.replace(".xls","")
    df.to_csv(f'data/preprocessed/{filename}_preprocessed.csv',index=False) # save dataframe in csv format
    return df

def preprocess_csv(dir, filename):
    from os import system
    from pandas import read_csv, Categorical

    df = read_csv(f'{dir}/{filename}') # load csv file into pandas dataframe
    cols = df.columns # get column names
    new_cols = [] # initialize list to contain new column names

    # remove time stamp column if it exists
    if(cols[0]=='Time Stamp'):
        df = df.drop([df.columns[0]], axis=1)
        print(f'{filename} Removed TimeStamp')
    cols = df.columns

    if(cols[0]=="Rodent Sleep"):
        for i in range(len(cols)): # deep copy column names to new column names
            new_cols.append(cols[i])
        for i in range(1,len(new_cols)-2): # for each column name, try to extract frequency range
            new_cols[i]=new_cols[i].split(',')[0][7:]
        for i in range(1,6): # cleanup for first 5 frequency ranges still containing " HZ"
            new_cols[i]=new_cols[i][0:5]
        print(new_cols[1])
        if(new_cols[1]!="0-0.5"):
            new_cols[1]="0-0.5"
        new_cols[-1]=new_cols[-1][:8] # remove end of column name
        new_cols[-2]=new_cols[-2][:5] # remove end of column name
        df.columns=new_cols # set dataframe column names to new column names
        df.rename(columns={"Rodent Sleep":"Class"},inplace=True)
        df.drop(df[df['Class'] == "X"].index, inplace = True)
        df["Class"]=Categorical(df["Class"]).fillna(method='backfill').codes # Convert to categorical codes here so we can analyze percentage of each class in next code block
        df = df.drop([0]) # drop first row containing units [muV^2]
        df = df.fillna(0) ## handle NaN values
        for col in df.loc[:, df.columns != 'Class']: # typecast each column to type float
            df[col] = df[col].astype(float)
    elif(cols[0]=="10 second Epochs"):
        del df[df.columns[0]] # remove first column
        for i in range(1,len(cols)): # deep copy column names to new column names
            new_cols.append(cols[i])
        for i in range(2,len(new_cols)-2): # for each column name, try to extract frequency range
            new_cols[i]=new_cols[i].split(',')[0][7:]
        for i in range(2,7): # cleanup for first 5 frequency ranges still containing " HZ"
            new_cols[i]=new_cols[i][0:5]
        new_cols[-2]=new_cols[-2][:8] # remove end of column name
        new_cols[-1]=new_cols[-1][:5] # remove end of column name
        df.columns=new_cols # set dataframe column names to new column names
        df.rename(columns={"Rodent Sleep":"Class"},inplace=True)
        df.drop(df[df['Class'] == "X"].index, inplace = True)
        df["Class"]=Categorical(df["Class"]).codes # Convert to categorical codes here so we can analyze percentage of each class in next code block
        df = df.drop([0]) # drop first row containing units [muV^2]
        df = df.fillna(0) ## handle NaN values
        for col in df.loc[:, df.columns != 'Class']: # typecast each column to type float
            df[col] = df[col].astype(float)
    else:
        for i in range(len(cols)): # deep copy column names to new column names
            new_cols.append(cols[i])    
        for i in range(0,len(new_cols)-2): # for each column name, try to extract frequency range
            new_cols[i]=new_cols[i].split(',')[0][7:]
        for i in range(0,5): # cleanup for first 5 frequency ranges still containing " HZ"
            new_cols[i]=new_cols[i][0:5]
        new_cols[-1]=new_cols[-1][:8] # remove end of column name
        new_cols[-2]=new_cols[-2][:5] # remove end of column name
        df.columns=new_cols # set dataframe column names to new column names
        df = df.drop([0]) # drop first row containing units [muV^2]
        df = df.fillna(0) ## handle NaN values
        for col in df.loc[:, df.columns != 'Class']: # typecast each column to type float
            df[col] = df[col].astype(float)
    system('mkdir data/preprocessed')
    filename = filename.replace(".csv","")
    df.to_csv("data/preprocessed/"+filename+"_preprocessed.csv",index=False) # save dataframe in csv format
    return df

def window_data(dir,target_filename):
    from tqdm import tqdm
    from pandas import read_csv,DataFrame
    from os import path,system
    df = read_csv(f'{dir}/{target_filename}')
    if ( not path.isdir('data/windowed')):
        system('mkdir data/windowed')
    filename = target_filename.replace(".csv",'')
    new_target_filename = "data/windowed/"+filename+"_windowed.csv"
    system('touch '+new_target_filename)

    for i in tqdm(range(len(df)-4)):
        win = df.iloc[i:i+5]
        x = win.values.flatten()
        X = DataFrame(x).T
        if i==0:
            X.to_csv(new_target_filename, mode='a', index=False)
        else:
            X.to_csv(new_target_filename, mode='a', index=False, header=False)

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
    if not os.isdir(new_dir):
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

def ZDBConversion(dir_csv, dir_zdb, csv, zdb, mode):
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