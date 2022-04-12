from posixpath import split
from subprocess import CalledProcessError
from lib.submodules import plot_metrics, train_model
TIME_DIR = ""

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
def print_yellow(str):
    print(f'{bcolors.WARNING}{str}{bcolors.ENDC}')
def print_green(str):
    print(f'{bcolors.OKGREEN}{str}{bcolors.ENDC}')
def print_red(str):
    print(f'{bcolors.FAIL}{str}{bcolors.ENDC}')

def rename_data_in_raw():
    print_yellow(f'Renaming data in raw')
    from os import listdir,system
    with open('data/mapping','w+') as f:
        system(f'mkdir data/renamed')
        for i,file in enumerate(listdir("data/raw")):
            f.write(f'{i},{file}\n')
            command = f'cp \"data/raw/{file}\" data/renamed/{str(i)}.xls'
            print_yellow(f"{i} {file}")
            print_yellow(command)
            system(command)
    print_green(f'Finished renaming data in raw')
def preprocess_renamed_files():
    print_yellow(f'Started preprocessing data')
    from os import listdir
    from lib.submodules import preprocess
    dir = f'data/renamed'
    for file in listdir(dir):
        print_yellow(file)
        preprocess(dir,file)
    print_green(f'Finished preprocessing data')
def fix_anomalies_in_preprocessed_files():
    print_yellow(f'Started fixing anomalies data')
    from os import listdir
    from lib.submodules import fix_anomalies
    dir = f'data/preprocessed'
    for file in listdir(dir):
        fix_anomalies(dir,file)
    print_green(f'Finished fixing anomalies data')
def select_features(select):
    print_yellow('Starting select features')
    from os import listdir
    from pandas import read_csv,DataFrame
    dir = f'data/preprocessed'
    temp = ['Class']
    select = temp + select
    for file in listdir(dir):
        print_yellow(dir+" "+file)
        df = read_csv(f'{dir}/{file}')
        Y = DataFrame()
        if(df.columns[0]!="Class"):
            print_red(f'Not scored, skipping')
            continue
        # Select features
        df = df[select]
        df.to_csv(f'{dir}/{file}',index=False)
    print_green('Finished select features')
def skip_features(skip):
    print_yellow('Starting skipping features')
    print_yellow(skip)
    from os import listdir
    from pandas import read_csv
    dir = 'data/preprocessed'
    for file in listdir(dir):
        print_yellow(dir+"/"+file)
        df = read_csv(f'{dir}/{file}')
        if(df.columns[0]!="Class"):
            print_red(f'Not scored, skipping')
            continue
        df = df.drop(columns=skip)
        df.to_csv(f'{dir}/{file}',index=False)
    print_green('Finished skipping features')
def window_preprocessed_files():
    print_yellow('Starting windowing')
    from os import listdir,system,path
    from lib.submodules import window
    if ( not path.isdir('data/windowed')):
        system('mkdir data/windowed')
    dir = f'data/preprocessed'
    for file in listdir(dir):
        print_yellow(file)
        window(dir,file)
    print_green('Finished windowing')
def balance_windowed_files():
    print_yellow('Starting balancing')
    from os import path,system,listdir
    from lib.submodules import balance
    dir = f'data/windowed'
    if (not path.isdir('data/balanced')):
        system(f'mkdir data/balanced')
    for file in listdir(dir):
        print_yellow(file)
        balance(dir,file)
    print_green('Finished balancing')
def concatenate_balanced_files():
    print_yellow('Starting Concatenating files')
    import pandas as pd
    import os
    from os import listdir
    from tqdm import tqdm
    filename = f'data/X.csv'
    for i,file in tqdm(enumerate(listdir("data/balanced"))):
        print_yellow(file)
        df = pd.read_csv("data/balanced/"+file)
        if(i==0):
            ## First, add header
            df.to_csv(filename, mode='w', header=True,index=False)
            continue
        df.to_csv(filename, mode='a', header=False,index=False)
    # new_filename = f'sessions/data/{TIME_DIR}/X.csv'
    # os.system(f'mkdir -p sessions/data/{TIME_DIR}')
    # os.system(f'cp {filename} {new_filename}')
    print_green('Finished Concatenating files')
def split_and_shuffle(dir=None):
    print_yellow('Starting split and shuffle')
    from pandas import read_csv
    import os

    os.system(f'mkdir -p sessions/data/{TIME_DIR}')
    if dir:
        os.system(f'cp sessions/data/{dir}/X.csv sessions/data/{TIME_DIR}/X.csv')      
    else:
        os.system(f'cp data/X.csv sessions/data/{TIME_DIR}/X.csv')      

    data_dir = f'sessions/data/{TIME_DIR}'
    df = read_csv(f'{data_dir}/X.csv')
    from sklearn.model_selection import train_test_split
    from numpy import array
    
    # Use a utility from sklearn to split and shuffle our dataset.
    train_df, test_df = train_test_split(df, test_size=0.2)
    train_df, val_df = train_test_split(train_df, test_size=0.2)
    train_df.to_csv(f"{data_dir}/train.csv",index=False)
    test_df.to_csv(f"{data_dir}/test.csv",index=False)
    val_df.to_csv(f"{data_dir}/val.csv",index=False)

    os.system(f'cp data/mapping {data_dir}/data_files.txt')
    os.system(f'cp {data_dir}/X.csv data/X.csv')
    os.system(f'cp {data_dir}/train.csv data/train.csv')
    os.system(f'cp {data_dir}/val.csv data/val.csv')
    os.system(f'cp {data_dir}/test.csv data/test.csv')

    # If split_and_shuffle was performed on data from another directory, update args.data_dir to train with new data
    if data_dir:
        return TIME_DIR
    else:
        return dir


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
    print_green('Finished split and shuffle')
def load_data_and_train_model(dir=None):
    print_yellow('Starting training model')
    from lib.submodules import train_model
    import pandas as pd
    import numpy as np
    import os
    os.system(f'mkdir -p sessions/models/{TIME_DIR}')
    if dir == None:
        data_dir = f'data'     # use test.csv, train.csv, and val.csv from this session
    else:
        data_dir = f'sessions/data/{dir}' # use test.csv, train.csv, and val.csv from previous session
    print_yellow(f'Using data in {data_dir}')
    train_df = pd.read_csv(f"{data_dir}/train.csv")
    val_df = pd.read_csv(f"{data_dir}/val.csv")
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
    hln = 512
    baseline_history = train_model(x_train,y_train,x_val,y_val,hln=hln)
    print_green('Finished training model')
    return hln,baseline_history

def load_data_and_test_model(hln, baseline_history,dir=None):
    print_yellow('Starting testing model')
    from lib.submodules import test_model,plot_cm
    from tensorflow import one_hot
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler

    if dir == None:
        data_dir = f'data'
    else:
        data_dir = f'sessions/data/{dir}'
    model_dir = f'sessions/models/{TIME_DIR}'
    test_df = pd.read_csv(f"{data_dir}/test.csv")
    y_test = test_df.pop('Class')
    x_test = test_df

    scaler = MinMaxScaler()
    x_test = scaler.fit_transform(x_test)

    x_test = np.array(x_test)
    y_test = np.array(y_test)
    

    baseline_results,test_predictions_baseline = test_model(x_test,y_test)
    plot_metrics(baseline_history)
    plot_cm(one_hot(y_test,depth=3).numpy().argmax(axis=1),test_predictions_baseline.argmax(axis=1),baseline_results,hln,"All Scored Files")
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
    plt.savefig(f"{model_dir}/cm.jpg")
    print_green('Finished testing model')
    return baseline_results

def upload_data(dir):
    """
    dir: name of rclone config (eg. 'drive')
    """
    import os
    import subprocess
    print_yellow('Starting Upload to Google Drive')
    args=['rclone', 'copy', f'sessions/models/{TIME_DIR}', f'{dir}:AuroraProject/sessions/{TIME_DIR}/' , '--drive-shared-with-me']
    try:
        subprocess.run(args, check=True)
    except CalledProcessError:
        print_red("Failed Upload to Google Drive")
        return
    # os.system(f'rclone copy sessions/models/{TIME_DIR} {dir}:AuroraProject/sessions/{TIME_DIR}/ --drive-shared-with-me')
    print_green('Finished Upload to Google Drive')

def create_time_dir():
    from datetime import datetime
    import os
    import subprocess
    now = datetime.now()
    date_str = now.strftime("%m.%d.%Y_%H:%M")
    global TIME_DIR 
    TIME_DIR = date_str
    print_yellow("Session: "+TIME_DIR)


def create_and_check_args():
    """
    Returns command line arguments in namespace 'args'
    """
    import os
    import argparse
    import subprocess

    parser = argparse.ArgumentParser(description='Pipeline to Train ANN Models')

    parser.add_argument('--new-data', required=False, action='store_true', dest='new_data',
                        help='Process new data located in "data/raw" (default: False)')
    parser.add_argument('--data-dir', metavar='MM.DD.YYYY_hh:mm', type=str, required=False, nargs='?', const=None, default=None, dest='data_dir',
                        help='If no new data, provide a directory located in "sessions/data/" to read data from (default: None)')
    parser.add_argument('--select-features', metavar='Feature', required=False, type=str, nargs='*', dest='select_features',
                        help=f"""Specify which features to {bcolors.BOLD}use{bcolors.ENDC} while training new model
                        [choices: "0-0.5", "0.5-1", ... , "19.5-20", "EEG2", "Activity"] (default: None) """)
    parser.add_argument('--skip-features', metavar='Feature', required=False, type=str, nargs='*', dest='skip_features',
                    help=f"""Specify which features to {bcolors.BOLD}skip{bcolors.ENDC} while training new model
                    [choices: "0-0.5", "0.5-1", ... , "19.5-20", "EEG2", "Activity"] (default: None) """)
    parser.add_argument('--split-and-shuffle', required=False, action='store_true', dest='do_split_shuffle', default=False,
                        help='Split and shuffle data (default: False)')
    parser.add_argument('--train-model', required=False, action='store_true', dest='do_train', default=False,
                        help='Train model from data (default: False)')
    parser.add_argument('--skip-upload', required=False, action='store_true', dest='skip_upload', default=False,
                        help='Skip uploading data to google drive (default: False)')
    parser.add_argument('--rclone-dir', required=False, metavar='rclone drive local name', type=str, default = None, dest='rclone_dir', nargs='?', const=None, 
                        help='Provide the name of your local rclone Google Drive directory containing the AuroraProject directory')
    args = parser.parse_args()
    if not args.new_data and not args.do_split_shuffle and not args.do_train:
        print_red("Must have at least one option selected\nrun ./main.py -h to see help")
        exit(1)
    if args.new_data:
        if args.do_train and not args.do_split_shuffle:
            print_red("Must specify --split-and-shuffle if using new data to train model\nrun ./main.py -h to see help")
            exit(1)
        if not os.path.isdir('data/raw'):
            print_red('Specified --new-data but no new data. Must add raw data to "data/raw"\nrun ./main.py -h to see help')
            exit(1)
        if args.data_dir:
            print_red("Cannot have options --new-data and --data-dir selected simultaneously\nrun ./main.py -h to see help")
            exit(1)
    if args.data_dir:
        if not os.path.isdir(f'sessions/data/{args.data_dir}'):
            print_red(f'sessions/data/{args.data_dir} does not exist\nrun ./main.py -h to see help')
            exit(1)
    if not args.new_data and not args.data_dir:
        if args.do_split_shuffle and not os.path.exists('data/X.csv'):
            print_red('data/X.csv does not exist, must specify --data-dir or process new data\nrun ./main.py -h to see help')
            exit(1)
        if args.do_train and (not os.path.exists('data/test.csv') or not os.path.exists('data/train.csv') or not os.path.exists('data/val.csv')): 
            print_red('Necessary data/*.csv files do not exist, must specify --split-and-shuffle to generate them or specify data directory to read data from (--data-dir)\nrun ./main.py -h to see help')
            exit(1)
    if args.select_features and args.skip_features:
        print_red("Cannot have options --skip-features and --select-features selected simultaneously\nrun ./main.py -h to see help")
        exit(1)
    if not args.new_data and (args.select_features or args.skip_features):
        print_red('Must use new data (--new-data) to skip or select features\nrun ./main.py -h to see help')
        exit(1)
    if not args.skip_upload and args.do_train:
        if not args.rclone_dir:
            print_red('If not skipping upload to google drive, must specify --rclone-dir to provide rclone Google Drive local name\nrun ./main.py -h to see help')
            exit(1)
        try:
            subprocess.run(["rclone", "lsd", f"{args.rclone_dir}:AuroraProject", "--drive-shared-with-me"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except CalledProcessError:
            print_red(f"{args.rclone_dir} is not a valid rclone config name or it does not contain 'AuroraProject' directory\nrun ./main.py -h to see help")
            exit(1)
    if args.new_data:
        print_yellow(f'Starting preprocessing with data in data/raw/')
    else:
        print_yellow(f'Training and testing with data in sessions/data/{args.data_dir}')
    return args