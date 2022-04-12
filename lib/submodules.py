def preprocess(dir,filename):
    from pandas import read_excel,Categorical
    from os import system
    file = "data/renamed/"+filename # filename
    df = read_excel(file) # load xls file into pandas dataframe
    cols = df.columns # get column names
    new_cols = [] # initialize list to contain new column names

    if(cols[0]=="Rodent Sleep"):
        for i in range(len(cols)): # deep copy column names to new column names
            new_cols.append(cols[i])
        for i in range(1,len(new_cols)-2): # for each column name, try to extract frequency range
            new_cols[i]=new_cols[i].split(',')[0][7:]
        for i in range(1,6): # cleanup for first 5 frequency ranges still containing " HZ"
            new_cols[i]=new_cols[i][0:5]
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
    df.to_csv("data/preprocessed/"+filename+"_preprocessed.csv",index=False) # save dataframe in csv format
    return df
def fix_anomalies(dir,file):
    from scripts.modules import print_yellow, print_red 
    import pandas as pd
    df = pd.read_csv(f'{dir}/{file}')
    print_yellow("======================================"+file)
    if(df.columns[0]!="Class"):
        print_red("NOT SCORED")
        return
    if(df.columns[1]!="0-0.5"):
        df.rename(columns={"EEG 1 (0-0.5 Hz, 0-0.5Hz , 10s) (Mean, 10s)":"0-0.5"},inplace=True)
    if("EEG 2" in df.columns):
        EEG_2 = df["EEG 2"]
    elif("EEG 1" in df.columns):
        EEG_1 = df["EEG 1"]
    Activity = df["Activity"]
    X = df.iloc[:,:-2]
    # TODO why is the second to last column EEG 1 or 2
    if("EEG 2" in df.columns):
        X.insert(X.shape[1],"EEG 2",EEG_2)
    elif("EEG 1" in df.columns):
        X.insert(X.shape[1],"EEG 1",EEG_1)
    X.insert(X.shape[1],"Activity",Activity)
    X.to_csv("data/preprocessed/"+file,index=False)
def window(dir,filename):
    from tqdm import tqdm
    from numpy import argmax,bincount,insert
    from pandas import read_csv,DataFrame,concat
    df = read_csv(f'{dir}/{filename}')
    Y = DataFrame()
    if(df.columns[0]!="Class"):
        return None
    for i in tqdm(range(len(df)-4)):
        win = df.iloc[i:i+5]
        c = argmax(bincount(win['Class']))
        del win['Class']
        x = win.values.flatten()
        x = insert(x,0,c)
        X = DataFrame(x).T
        X = X.rename({0: 'Class'}, axis='columns')
        Y = concat([Y,X])
    df_win = Y
    df_win = df_win.reset_index()
    del df_win['index']
    df_win['Class'] = df_win['Class'].astype(int)
    df = df_win
    filename = filename.replace(".csv","")
    df.to_csv("data/windowed/"+filename+"_windowed.csv",index=False)
    return df
def balance(dir,filename):
    from pandas import read_csv,concat
    from lib.utils import class_count
    df = read_csv(f'{dir}/{filename}')
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
        df = concat([df,locs[min_ind]])
    p,s,w = class_count(df)

    filename = filename.replace(".csv","")
    df.to_csv("data/balanced/"+filename+"_balanced.csv",index=False)
    return df
def train_model(x_train,y_train,x_val,y_val,hln=256):
    import tensorflow as tf
    from tensorflow import keras
    BATCH_SIZE=64
    EPOCHS=512
    model = tf.keras.Sequential([
        keras.layers.Dense(hln, activation='relu',input_shape=(x_train.shape[-1],)),
        keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                metrics=[
        keras.metrics.CategoricalAccuracy(name='categorical_accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc')
    ])

    from lib.modules import TIME_DIR
    model_dir = f"sessions/models/{TIME_DIR}"

    from time import time
    start = time()
    baseline_history = model.fit(
        x_train,
        tf.one_hot(y_train, 3),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_val, tf.one_hot(y_val, depth=3)),
        callbacks=[keras.callbacks.ModelCheckpoint(
            f"{model_dir}/best_model.h5", save_best_only=True, monitor="val_loss",verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
        ),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1,mode='min',restore_best_weights=True),])
    stop = time()-start
    return baseline_history
def test_model(x_test,y_test):
    from keras.models import load_model
    from lib.modules import TIME_DIR
    # data_dir = f'sessions/data/{TIME_DIR}'
    model_dir = f"sessions/models/{TIME_DIR}"
    model = load_model(f"{model_dir}/best_model.h5")
    from sklearn.preprocessing import MinMaxScaler

    # from scripts.utils import *
    # plt.rcParams["figure.facecolor"] = 'w'
    # plot_metrics(baseline_history,"",hln)
    import tensorflow as tf
    test_predictions_baseline = model.predict(x_test, batch_size=64)
    baseline_results = model.evaluate(x_test, tf.one_hot(y_test,depth=3),
                                        batch_size=64, verbose=0)

    return baseline_results,test_predictions_baseline
def plot_metrics(baseline_history):
    ## Plot Metrics Save as PGF
    import matplotlib as mpl
    from matplotlib import pyplot as plt
    history = baseline_history
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    mpl.rcParams['figure.figsize'] = (7.2,4.45)
    metrics = ['loss', 'categorical_accuracy','precision','recall']
    metric = metrics[1]
    name = metric.replace("_"," ").capitalize()
    plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
    plt.plot(history.epoch, history.history['val_'+metric],
                color=colors[1], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel(name)
    plt.title(metric)
    if metric == 'loss':
        plt.ylim([0, plt.ylim()[1]])
    elif metric == 'auc':
        plt.ylim([0.8,1])
    else:
        plt.ylim([0,1])
    plt.legend()
    from lib.modules import TIME_DIR
    model_dir = f"sessions/models/{TIME_DIR}"
    plt.savefig(f"{model_dir}/{metric}.jpg",bbox_inches='tight',dpi=200)
def plot_cm(labels, predictions,met,hln,file,save=True,model_dir='.'):
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    plt.figure()
    cm = confusion_matrix(labels, predictions)
    sns.heatmap(cm, annot=True, fmt="d",cbar=False)
    plt.title('Confusion Matrix for '+file+'\nloss: '+str(met[0])+'\nacc: '+str(met[1])+'\nhidden layer: '+str(hln))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    if(save):
        plt.savefig(f"{model_dir}/cm.jpg",bbox_inches='tight',dpi=200)


# def sub_test():
#     from scripts.modules import TIME_DIR
#     print(TIME_DIR)


