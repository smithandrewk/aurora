import pandas as pd 
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

def load_xls(filename):
    """
    load any generic xls file from data folder into pandas dataframe
    """
    file = "data/"+filename+".xls" # filename
    return pd.read_excel(file) # load xls file into pandas dataframe
def preprocess(filename):
    """
    First I obtain data into a pandas dataframe from .xls format given by SOM. 
    control.xls has shape (8641,43) where the first column is the targets and the
    following 42 columns are the features. Most of the features are EEG channels,
    but there is one feature which is "activity".
    """
    df = load_xls(filename)
    """
    clean up this specific formatting of data from UofSCSOM
    """
    cols = df.columns # get column names
    new_cols = [] # initialize list to contain new column names
    for i in range(len(cols)): # deep copy column names to new column names
        new_cols.append(cols[i])
    for i in range(2,len(new_cols)-2): # for each column name, try to extract frequency range
        new_cols[i]=new_cols[i].split(',')[0][7:]
    for i in range(2,7): # cleanup for first 5 frequency ranges still containing " HZ"
        new_cols[i]=new_cols[i][0:5]
    new_cols[42]=new_cols[42][:8] # remove end of column name
    new_cols[43]=new_cols[43][:5] # remove end of column name
    df.columns=new_cols # set dataframe column names to new column names
    df = df.drop([0]) # drop first row containing units [muV^2]
    del df[df.columns[0]] # remove first column
    """
    There are some NaN values in the dataframe. I handle by filling them in with
    zeros. I think we might transition to another method such as mean.
    """
    df = df.fillna(0) ## handle NaN values
    """
    Rename "Rodent Sleep" column to "Class"
    """
    df.rename(columns={"Rodent Sleep":"Class"},inplace=True)
    """
    We have a set of targets which is {'P','S','W'} and we convert this set into a
    categorical data type with pd.Categorical(y). This gives a categorical array
    values from the same set. pd.Categorical(y).codes gives the categorical array
    after the mapping from the set {'P','S','W'} to {0,1,2}.
    """
    df["Class"]=pd.Categorical(df["Class"]).codes # Convert to categorical codes here so we can analyze percentage of each class in next code block
    """
    If we check df.values, we will notice that each entry is of type "object"; thus,
    we typecast the entries as floats.
    """
    for col in df.loc[:, df.columns != 'Class']: # typecast each column to type float
        df[col] = df[col].astype(float)
    df.to_csv("data/control_preprocessed.csv",index=False) # save dataframe in csv format
    return df
def plot_cm(labels, predictions, p=0.5):
  cm = confusion_matrix(labels, predictions > p)
  plt.figure(figsize=(5,5))
  sns.heatmap(cm, annot=True, fmt="d")
  plt.title('Confusion matrix @{:.2f}'.format(p))
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')