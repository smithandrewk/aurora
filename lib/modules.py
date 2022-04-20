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

def rename_data_in_raw():
    import os
    print_yellow("Starting renaming data in raw")
    os.system('mkdir -p data/renamed')
    f = open('data/mapping','w+')
    for i, file in enumerate(os.listdir("data/raw")):
        original_name = file
        file = file.replace(" ", "\ ")
        new_name = str(i)  + ".csv"
        cmd = f"ssconvert data/raw/{file} data/renamed/{new_name}"
        os.system(cmd)
        print_yellow(f"Iteration {i}: Converting {original_name}")
        f.write(original_name+'\n')
    f.close()
    print_green("Finished Renaming")
def initial_preprocessing():
    """
    initial_preprocessing does.

    @params
        filename : name of file
    """
    print_yellow('Starting Preprocessing')
    import os
    from lib.submodules import preprocess_csv

    i = 0
    dir = f'data/renamed'
    for file in os.listdir(dir):
        print_yellow("Iteration " + str(i))
        print_yellow(dir+" "+file)
        preprocess_csv(dir,file)
        i += 1
    print_green("Finishing Preprocessing")
def handle_anomalies():
    """
    handle_anomalies handles anomalies.

    @params
        filename : name of file
    """
    print_yellow("Starting Handling Anomalies")
    import pandas as pd
    from os import listdir
    for file in listdir("data/preprocessed"):
        df = pd.read_csv("data/preprocessed/"+file)
        print_yellow("======================================"+file)
        if(df.columns[0]!="Class"):
            print_yellow("NOT SCORED")
            continue
        if(df.columns[1]!="0-0.5"):
            print(f"{bcolors.FAIL}ANOMALY{bcolors.ENDC}")
            df.rename(columns={"EEG 1 (0-0.5 Hz, 0-0.5Hz , 10s) (Mean, 10s)":"0-0.5"},inplace=True)
        EEG_2 = df["EEG 2"]
        Activity = df["Activity"]
        X = df.iloc[:,:-2]
        X.insert(X.shape[1],"EEG 2",EEG_2)
        X.insert(X.shape[1],"Activity",Activity)
        X.to_csv("data/preprocessed/"+file,index=False)

    print_green("Finishing Handling Anomalies")
def window():
    """
    window windows.

    @params
        filename : name of file
    """
    print_yellow("Starting Windowing")
    from lib.submodules import window_data
    from os import listdir
    i = 0
    dir = f'data/preprocessed'
    for file in listdir(dir):
        print_yellow("Iteration: " + str(i))
        print_yellow(file)
        # X = read_csv(i)
        window_data(dir,file)
        i += 1
    print_green("Finishing Windowing")
def scale():
    """
    scale scales.

    @params
        filename : name of file
    """
    print_yellow("Starting Scaling")
    import pandas as pd
    import os
    i = 0
    dir = 'data/windowed'
    for file in os.listdir(dir):
        filename = f'{dir}/{file}'
        X = pd.read_csv(filename)
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        print_yellow("Iteration: " + str(i))
        X = scaler.fit_transform(X)
        if ( not os.path.isdir('data/windowed_scaled')):
            os.system('mkdir data/windowed_scaled')
        file = file.replace("_preprocessed_windowed.csv", "")
        filename = "data/windowed_scaled/"+file+"_windowed_scaled.csv"
        pd.DataFrame(X).to_csv(filename, index=False)
        print_yellow(filename)
        i += 1
    print_green("Finishing Scaling")
def score_ann(model):
    """
    score_ann scores ANN.

    @params
        model : name of model
    """
    print_yellow("Started Scoring ANN")
    from lib.submodules import score_data_ann
    from os import listdir

    dir = 'data/windowed_scaled'
    i=0
    for file in listdir(dir):
        print_yellow("Iteration: " + str(i))
        score_data_ann(dir, file, model)
        i += 1

    print_green("Finishing Scoring ANN")
def score_rf(model):
    """
    score_rf scores RF.

    @params
        model : name of model
    """
    print_yellow("Starting Scoring RF")
    from lib.submodules import score_data_rf
    from os import listdir

    dir = 'data/windowed'
    i=0
    for file in listdir(dir):
        print_yellow("Iteration: " + str(i))
        score_data_rf(dir, file, model)
        i+=1

    print_green("Finishing Scoring RF")
def expand_predictions():
    """
    expand_prediction expands predictions.

    @params
        filename : name of file
    """
    print_yellow("Starting Expand Predictions")
    from lib.submodules import expand_predictions_ann, expand_predictions_rf
    from os import listdir

    dir_ann = 'data/predictions_ann'
    i=0
    print_yellow("Expanding ANN predictions")
    for file in listdir(dir_ann):
        print("Iteration: " + str(i))
        expand_predictions_ann(dir_ann, file)
        i+=1
    dir_rf = 'data/predictions_rf'
    print_yellow("Expanding RF predictions")
    i=0
    for file in listdir(dir_rf):
        print_yellow("Iteration: " + str(i))
        expand_predictions_rf(dir_rf, file)
        i+=1

    print_green("Finishing Expand Predictions")
def rename_scores():
    """
    rename_scores renames scores (0,1,2 --> P,S,W)
    """
    print_yellow("Starting Rename Scores")
    import os
    import pandas as pd
    from pandas import read_csv
    rename_dict = {0: 'P', 1: 'S', 2: 'W'}
    # rename_dict = {0: 'Sleep-Paradoxical', 1: 'Sleep-SWS', 2: 'Sleep-Wake'}  #doesnt work with matlabcode
    if ( not os.path.isdir('data/expanded_renamed_rf')):
        os.system('mkdir data/expanded_renamed_rf')
    if ( not os.path.isdir('data/expanded_renamed_ann')):
        os.system('mkdir data/expanded_renamed_ann')
    
    dir_ann = 'data/expanded_predictions_ann'
    for file in os.listdir(dir_ann):
        df = read_csv(f'{dir_ann}/{file}')
        y = df['0']
        new_y = []
        for i in y:
            new_y.append(rename_dict[i])
        pd.DataFrame(new_y).to_csv("data/expanded_renamed_ann/"+file,index=False)
    
    dir_rf = 'data/expanded_predictions_rf'
    for file in os.listdir(dir_rf):
        df = read_csv(f'{dir_rf}/{file}')
        y = df['0']
        new_y = []
        for i in y:
            new_y.append(rename_dict[i])
        pd.DataFrame(new_y).to_csv("data/expanded_renamed_rf/"+file,index=False)
    
    print_green("Finishing Rename Scores")
def remap_names():
    """
    remap_names remaps names to original names

    @params
        filename : name of file
    """
    print_yellow("Starting Remap Names")
    import os
    mapping = open('data/mapping').read().splitlines()
    if not os.path.isdir('data/final_ann'):
        os.system('mkdir data/final_ann')
    if not os.path.isdir('data/final_rf'):
        os.system('mkdir data/final_rf')
    i=0
    for file in os.listdir('data/expanded_renamed_ann'):
        index_str = file.replace('.csv', '')
        newName = mapping[int(index_str)].replace('.xls', '-ann.csv')
        os.system(f"cp data/expanded_renamed_ann/'{file}' data/final_ann/'{newName}'")
        i+=1
    i=0
    for file in os.listdir('data/expanded_renamed_rf'):
        index_str = file.replace('.csv', '')
        newName = mapping[int(index_str)].replace('.xls', '-rf.csv')
        os.system(f"cp data/expanded_renamed_rf/'{file}' data/final_rf/'{newName}'")
        i+=1
    print_green("Finishing Remap Names")
def zdb_preprocess():
    """
    zdb_preprocess preprocesses ZDBs
    """
    print_yellow("Starting ZDB preprocessing")
    import os
    from lib.submodules import preprocess_zdb
    dir = 'data/renamedZDB'
    if not os.path.isdir(dir):
        print("No ZDB files")
        return
    for file in os.listdir(dir):
        preprocess_zdb(dir, file)
    print_green("Finishing ZDB preprocessing")

def zdb_conversion():
    """
    zdb_conversion imports csv into ZDB format.
    """
    print_yellow("Starting ZDB Conversion")
    import os
    from lib.submodules import conversion_zdb

    dir_zdb = 'data/preprocessedZDB'
    dir_ann = 'data/expanded_renamed_ann'
    dir_rf = 'data/expanded_renamed_rf'

    if not os.path.isdir(dir_zdb):
        print("No ZDB files")
        return
    for csv in os.listdir(dir_ann):
        name = csv.replace('.csv', '')
        zdb = f'{name}.zdb'
        conversion_zdb(dir_ann, dir_zdb, csv, zdb, 'ann')
    for csv in os.listdir(dir_rf):
        name = csv.replace('.csv', '')
        zdb = f'{name}.zdb'
        conversion_zdb(dir_ann, dir_zdb, csv, zdb, 'rf')
    print_green("Finishing ZDB Conversion")
def zdb_remap():
    """
    zdb_remap remaps zdb names
    """
    print_yellow("Starting ZDB remap names")
    import os
    dir_ann = "data/ZDB_ann"
    dir_rf = "data/ZDB_rf"
    if not os.path.isdir(dir_ann) or not os.path.isdir(dir_rf):
        print("No ZDB files")
        return
    os.system('mkdir data/ZDB_final_ann')
    os.system('mkdir data/ZDB_final_rf')
    mapping = open('data/ZDBmapping').read().splitlines()

    for zdb in os.listdir(dir_ann):
        index = int(zdb.replace('.zdb', ''))
        os.system(f'cp {dir_ann}/"{zdb}" data/ZDB_final_ann/"{mapping[index]}"')
    for zdb in os.listdir(dir_rf):
        index = int(zdb.replace('.zdb', ''))
        os.system(f'cp {dir_rf}/"{zdb}" data/ZDB_final_rf/"{mapping[index]}"')
    print_green("Finished ZDB remapping")

def create_and_check_args():
    import argparse
    import os
    parser = argparse.ArgumentParser(description='Pipeline to Score Data')
    parser.add_argument('--ann-model', type=str, required=True, dest='ann_model', metavar='[path-to-ann-model]',
                        help=f'Enter path to ANN model within {bcolors.BOLD}model{bcolors.ENDC} directory')
    parser.add_argument('--rf-model', type=str, required=True, dest='rf_model', metavar='[path-to-rf-model]',
                        help=f'Enter path to RF model within {bcolors.BOLD}model{bcolors.ENDC} directory')
    args = parser.parse_args()
    if not os.path.exists('model/'+args.ann_model):
        print(f"{bcolors.FAIL}Invalid path to ANN model{bcolors.ENDC}")
        exit(1)
    if not os.path.exists('model/'+args.rf_model):
        print(f"{bcolors.FAIL}Invalid path to RF model{bcolors.ENDC}")
        exit(1)
    print_yellow(f'Using ANN model: model/{args.ann_model}')
    print_yellow(f'Using RF model: model/{args.rf_model}')
    return args