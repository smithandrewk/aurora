import pandas as pd
from pandas.core.indexes import base
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow import keras
import pandas as pd 
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow import keras
from IPython.display import clear_output
from time import strftime
from tqdm import tqdm
import os.path
from os import path
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from time import time

from tensorflow import keras

from tensorflow.keras.callbacks import TensorBoard
from scripts.utils import *
import sklearn
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from time import strftime
def preprocess(filename):
    file = "data/renamed/"+filename # filename
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
        print(new_cols[1])
        if(new_cols[1]!="0-0.5"):
            new_cols[1]="0-0.5"
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

def plot_metrics(history,date,hln):
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
    plt.savefig("metrics.png",bbox_inches='tight',transparent=False)
    # plt.savefig("figures/"+str(date[1])+"@"+str(date[0][:5].replace(":",""))+"_"+str(hln)+"neurons_training_metrics.png",bbox_inches='tight',transparent=False)
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
        tf.keras.layers.LSTM(100, input_shape=(10, 211)),
        keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(3, activation='sigmoid')
        ])
    else:
        model = tf.keras.Sequential([
        keras.layers.Dense(n, activation='relu',input_shape=INPUT_FEATURES),
        tf.keras.layers.LSTM(100), 
        tf.keras.layers.Dense(3, activation='sigmoid')
        ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                metrics=[
      keras.metrics.CategoricalAccuracy(name='categorical_accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc')
      ])
    #
    model.summary()
    return model
def train(train_features,train_labels,val_features,val_labels,test_features,test_labels,class_weight,INPUT_FEATURES,hln=20,EPOCHS=100,BATCH_SIZE=200,weights=False,):
    model = get_compiled_model(hln,INPUT_FEATURES=INPUT_FEATURES,dropout=True)
    """
    We one-hot encode the targets. Mathematically, this is good for calculating
    loss. CategoricalCrossEntropy simplifies to a negative log when targets are
    one-hot encoded. However, I simply recieved an error from model.fit when I 
    did not one-hot encode.
      @y : targets
      @depth : number of targets
    """
    # Callback choices
    plot_losses = TrainingPlot()
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='categorical_accuracy', 
        verbose=1,
        patience=100,
        mode='max',
        restore_best_weights=True)
    # tensorboard = TensorBoard(log_dir='logs/{}'.format(time()))
    if(weights):
        baseline_history = model.fit(
            train_features,
            tf.one_hot(train_labels,depth=3),
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(val_features, tf.one_hot(val_labels,depth=3)),
            callbacks=[early_stopping],
            class_weight=class_weight)
    else:
        baseline_history = model.fit(
        train_features,
        tf.one_hot(train_labels,depth=3),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(val_features, tf.one_hot(val_labels,depth=3)),
        callbacks=[early_stopping])
    train_predictions_baseline = model.predict(train_features, batch_size=BATCH_SIZE)
    test_predictions_baseline = model.predict(test_features, batch_size=BATCH_SIZE)
    baseline_results = model.evaluate(test_features, tf.one_hot(test_labels,depth=3),
                                      batch_size=BATCH_SIZE, verbose=0)
    print(baseline_results[1])
    plot_cm(tf.one_hot(test_labels,depth=3).numpy().argmax(axis=1),test_predictions_baseline.argmax(axis=1),baseline_results,hln,"all")
    date = strftime('%X %x').replace("/","").split()
    plt.savefig("figures/"+str(date[1])+"@"+str(date[0][:5].replace(":",""))+"_"+str(hln)+"neurons_confusion_matrix.png",bbox_inches='tight')
    return baseline_history,baseline_results,date,model
def split_and_shuffle(df):
    p,s,w = class_count(df)

    # Use a utility from sklearn to split and shuffle our dataset.
    train_df, test_df = train_test_split(df, test_size=0.2)
    train_df, val_df = train_test_split(train_df, test_size=0.2)
    train_df.to_csv("train.csv",index=False)
    test_df.to_csv("test.csv",index=False)
    val_df.to_csv("val.csv",index=False)

    # Form np arrays of labels and features.
    train_labels = np.array(train_df.pop('Class'))
    p_train_labels = train_labels == 0
    s_train_labels = train_labels == 1
    w_train_labels = train_labels == 2

    val_labels = np.array(val_df.pop('Class'))
    test_labels = np.array(test_df.pop('Class'))

    train_features = np.array(train_df)
    val_features = np.array(val_df)
    test_features = np.array(test_df)
    total = p + s + w
    # Scaling by total/2 helps keep the loss to a similar magnitude.
    # The sum of the weights of all examples stays the same.
    weight_for_p = (1 / p)*(total)/2.0 
    weight_for_w = (1 / w)*(total)/2.0
    weight_for_s = (1 / s)*(total)/2.0


    class_weight = {0: weight_for_p, 1: weight_for_s, 2: weight_for_w}

    print('Weight for class 0: {:.2f}'.format(weight_for_p))
    print('Weight for class 1: {:.2f}'.format(weight_for_s))
    print('Weight for class 2: {:.2f}'.format(weight_for_w))

    return train_features,train_labels,val_features,val_labels,test_features,test_labels,class_weight