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