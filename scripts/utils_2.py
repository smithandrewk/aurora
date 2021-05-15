import pandas as pd
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


def preprocess(filename):
    file = "data/test/"+filename # filename
    df = pd.read_excel(file) # load xls file into pandas dataframe
    cols = df.columns # get column names
    new_cols = [] # initialize list to contain new column names

    if(cols[0]=="Rodent Sleep"):
        for i in range(len(cols)): # deep copy column names to new column names
            new_cols.append(cols[i])
        for i in range(1,len(new_cols)-2): # for each column name, try to extract frequency range
            new_cols[i]=new_cols[i].split(',')[0][7:]
        for i in range(1,6): # cleanup for first 5 frequency ranges still containing " HZ"
            new_cols[i]=new_cols[i][0:5]
        new_cols[-1]=new_cols[-1][:8] # remove end of column name
        new_cols[-2]=new_cols[-2][:5] # remove end of column name
        df.columns=new_cols # set dataframe column names to new column names
        df.rename(columns={"Rodent Sleep":"Class"},inplace=True)
        df.drop(df[df['Class'] == "X"].index, inplace = True)
        df["Class"]=pd.Categorical(df["Class"]).fillna(method='backfill').codes # Convert to categorical codes here so we can analyze percentage of each class in next code block
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
        df["Class"]=pd.Categorical(df["Class"]).codes # Convert to categorical codes here so we can analyze percentage of each class in next code block
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
    os.system('mkdir data/preprocessed')
    filename = filename.replace(".xls","")
    df.to_csv("data/preprocessed/"+filename+"_preprocessed.csv",index=False) # save dataframe in csv format
    return df
def window(target_filename):
    df = pd.read_csv("data/preprocessed/"+target_filename)
    Y = pd.DataFrame()
    if(df.columns[0]!="Class"):
        return None
    for i in tqdm(range(len(df)-4)):
        win = df.iloc[i:i+5]
        c = np.argmax(np.bincount(win['Class']))
        del win['Class']
        x = win.values.flatten()
        x = np.insert(x,0,c)
        X = pd.DataFrame(x).T
        X = X.rename({0: 'Class'}, axis='columns')
        Y = pd.concat([Y,X])
    df_win = Y
    df_win = df_win.reset_index()
    del df_win['index']
    df_win['Class'] = df_win['Class'].astype(int)
    df = df_win
    if ( not os.path.isdir('data/windowed')):
        os.system('mkdir data/windowed')
    target_filename = target_filename.replace(".csv","")
    df.to_csv("data/windowed/"+target_filename+"_windowed.csv",index=False)
    return df
def balance(target_filename):
    df = pd.read_csv("data/windowed/"+target_filename)
    X = [p,s,w] = class_count(df)
    min_val = min(X)
    min_ind = X.index(min_val)
    max_val = max(X)
    max_ind = X.index(max_val)
    # ## Balancing
    # # TODO : balancing algorithm
    ps = df.loc[df["Class"]==0]
    ss = df.loc[df["Class"]==1]
    ws = df.loc[df["Class"]==2]
    locs = [ps,ss,ws]
    for i in range(int(max_val/min_val)):
        df = pd.concat([df,locs[min_ind]])
    p,s,w = class_count(df)
    if ( not os.path.isdir('data/windowed_balanced')):
        os.system('mkdir data/windowed_balanced')
    target_filename = target_filename.replace(".csv","")
    df.to_csv("data/windowed_balanced/"+target_filename+"_balanced.csv",index=False)
    return df
def class_count(df):
    p,s,w = np.bincount(df['Class'])
    total = p + s + w
    print('Examples:\n    Total: {}\n    P: {} ({:.2f}% of total)\n    S: {} ({:.2f}% of total)\n    W: {} ({:.2f}% of total)\n'.format(
        total, p, 100 * p / total,s,100 * s / total,w,100 * w / total))
    return p,s,w
def plot_cm(labels, predictions,met,hln,file):
    plt.figure()
    cm = confusion_matrix(labels, predictions)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion Matrix for '+file+'\nloss: '+str(met[0])+'\nacc: '+str(met[1])+'\nhidden layer: '+str(hln))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')