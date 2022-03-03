from submodules import plot_metrics, train_model


def rename_data_in_raw():
    print(f'Renaming data in raw')
    from os import listdir,system
    with open('data/mapping','w+') as f:
        system(f'mkdir data/renamed')
        for i,file in enumerate(listdir("data/raw")):
            f.write(f'{i},{file}\n')
            command = f'cp \"data/raw/{file}\" data/renamed/{str(i)}.xls'
            print(i,file)
            print(command)
            system(command)
def preprocess_renamed_files():
    from os import listdir
    from submodules import preprocess
    dir = f'data/renamed'
    for file in listdir(dir):
        preprocess(dir,file)
def fix_anomalies_in_preprocessed_files():
    from os import listdir
    from submodules import fix_anomalies
    dir = f'data/preprocessed'
    for file in listdir(dir):
        fix_anomalies(dir,file)
def window_preprocessed_files():
    from os import listdir,system,path
    from submodules import window
    if ( not path.isdir('data/windowed')):
        system('mkdir data/windowed')
    dir = f'data/preprocessed'
    for file in listdir(dir):
        window(dir,file)
def balance_windowed_files():
    from os import path,system,listdir
    from submodules import balance
    dir = f'data/preprocessed'
    if (not path.isdir('data/balanced')):
        system(f'mkdir data/balanced')
    for file in listdir(dir):
        balance(dir,file)
def concatenate_balanced_files():
    import pandas as pd
    from os import listdir
    from tqdm import tqdm

    for i,file in tqdm(enumerate(listdir("data/balanced"))):
        df = pd.read_csv("data/balanced/"+file)
        if(i==0):
            ## First, add header
            df.to_csv('X.csv', mode='w', header=True,index=False)
            continue
        df.to_csv('X.csv', mode='a', header=False,index=False)
def split_and_shuffle(filename):
    from pandas import read_csv
    df = read_csv(filename)
    from sklearn.model_selection import train_test_split
    from numpy import array
    # Use a utility from sklearn to split and shuffle our dataset.
    train_df, test_df = train_test_split(df, test_size=0.2)
    # train_df, val_df = train_test_split(train_df, test_size=0.2)
    train_df.to_csv("train.csv",index=False)
    test_df.to_csv("test.csv",index=False)
    # val_df.to_csv("val.csv",index=False)

    # Form np arrays of labels and features.
    # train_labels = array(train_df.pop('Class'))
    # p_train_labels = train_labels == 0
    # s_train_labels = train_labels == 1
    # w_train_labels = train_labels == 2

    # val_labels = array(val_df.pop('Class'))
    # test_labels = array(test_df.pop('Class'))

    # train_features = array(train_df)
    # val_features = array(val_df)
    # test_features = array(test_df)
    # total = p + s + w
    # Scaling by total/2 helps keep the loss to a similar magnitude.
    # The sum of the weights of all examples stays the same.
    # weight_for_p = (1 / p)*(total)/2.0 
    # weight_for_w = (1 / w)*(total)/2.0
    # weight_for_s = (1 / s)*(total)/2.0


    # class_weight = {0: weight_for_p, 1: weight_for_s, 2: weight_for_w}

    # print('Weight for class 0: {:.2f}'.format(weight_for_p))
    # print('Weight for class 1: {:.2f}'.format(weight_for_s))
    # print('Weight for class 2: {:.2f}'.format(weight_for_w))
def load_data_and_train_model():
    from submodules import train_model
    import pandas as pd
    import numpy as np
    train_df = pd.read_csv("train.csv")
    val_df = pd.read_csv("val.csv")
    y_train = train_df.pop('Class')
    x_train = train_df
    y_val = val_df.pop('Class')
    x_val = val_df
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.fit_transform(x_val)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_val = np.array(x_val)
    y_val = np.array(y_val)
    baseline_history = train_model(x_train,y_train,x_val,y_val)
    return baseline_history
def load_data_and_test_model():
    from submodules import test_model,plot_cm
    from tensorflow import one_hot
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    test_df = pd.read_csv("test.csv")
    y_test = test_df.pop('Class')
    x_test = test_df
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    x_test = scaler.fit_transform(x_test)

    baseline_results,test_predictions_baseline = test_model(x_test,y_test)
    # plot_metrics(baseline_history)
    plot_cm(one_hot(y_test,depth=3).numpy().argmax(axis=1),test_predictions_baseline.argmax(axis=1),baseline_results,512,"All Scored Files")
    # import matplotlib
    # matplotlib.use("pgf")
    # plt.style.use("style.txt")
    # matplotlib.rcParams.update({
    #     "pgf.texsystem": "xelatex",
    #     'font.family': 'serif',
    #     'text.usetex': True,
    #     'pgf.rcfonts': False
    # })
    plt.show()
    plt.savefig("cm.jpg")
    return baseline_results
