import pandas as pd 
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from IPython.display import clear_output
from time import time
from time import strftime
from tqdm import tqdm
import os.path
from os import path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from time import time
import sklearn
from sklearn.model_selection import train_test_split

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
    df.to_csv("data/"+filename+"_preprocessed.csv",index=False) # save dataframe in csv format
    return df
def plot_cm(labels, predictions,met,hln,file):
    plt.figure()
    cm = confusion_matrix(labels, predictions)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion Matrix for '+file+'\nloss: '+str(met[0])+'\nacc: '+str(met[1])+'\nhidden layer: '+str(hln))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
def class_count(df):
    p,s,w = np.bincount(df['Class'])
    total = p + s + w
    print('Examples:\n    Total: {}\n    P: {} ({:.2f}% of total)\n    S: {} ({:.2f}% of total)\n    W: {} ({:.2f}% of total)\n'.format(
        total, p, 100 * p / total,s,100 * s / total,w,100 * w / total))
    return p,s,w
def get_compiled_model(n,INPUT_FEATURES,dropout=True):
    """
    Function to create model. This is a sequential model, meaning layers execute
    one after the other. We have an input shape corresponding to the feature set,
    one hidden layer with 10 neurons and a relu activation, then an output layer
    with 3 neurons and a sigmoid activation function. We compute loss with
    categorical crossentropy and optimize with adam, which I believe is something
    about an adaptive learning rate. I do not know what the parameter from_logits
    is about.
    """
    if(dropout):
        model = tf.keras.Sequential([
        keras.layers.Dense(n, activation='relu',input_shape=INPUT_FEATURES),
        keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(3, activation='sigmoid')
        ])
    else:
        model = tf.keras.Sequential([
        keras.layers.Dense(n, activation='relu',input_shape=INPUT_FEATURES),
        tf.keras.layers.Dense(3, activation='sigmoid')
        ])
    model.compile(optimizer=keras.optimizers.Adam(lr=1e-3),
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                metrics=[
      keras.metrics.CategoricalAccuracy(name='categorical_accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
])
    return model
class TrainingPlot(keras.callbacks.Callback):
    
    # This function is called when the training begins
    def on_train_begin(self, logs={}):
        # Initialize the lists for holding the logs, losses and accuracies
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []
    
    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
        
        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('categorical_accuracy'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_categorical_accuracy'))
        
        # Before plotting ensure at least 2 epochs have passed
        if len(self.losses) > 1:
            
            # Clear the previous plot
            clear_output(wait=True)
            N = np.arange(0, len(self.losses))
            
            # You can chose the style of your preference
            # print(plt.style.available) to see the available options
            plt.style.use("seaborn")
            
            # Plot train loss, train acc, val loss and val acc against epochs passed
            plt.figure(figsize=(20,10))
            plt.subplot(1,2,1)
            plt.plot(N, self.losses, label = "train_loss")
            plt.plot(N, self.val_losses, label = "val_loss")
            plt.subplot(1,2,2)
            plt.plot(N, self.acc, label = "train_acc")
            plt.plot(N, self.val_acc, label = "val_acc")
            plt.title("Training Loss and Accuracy [Epoch {}]".format(epoch))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            plt.show()
def plot_metrics(history):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    mpl.rcParams['figure.figsize'] = (12, 10)
    metrics = ['loss', 'categorical_accuracy','precision','recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(2,2,n+1)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric],
                 color=colors[1], label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8,1])
        else:
            plt.ylim([0,1])
        plt.legend()