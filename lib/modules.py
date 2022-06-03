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
    print_yellow(f'Renaming data in raw')
    from os import listdir, system
    with open('data/mapping', 'w+') as f:
        system(f'mkdir data/1_renamed')
        for i, file in enumerate(listdir("data/0_raw")):
            f.write(f'{i},{file}\n')
            command = f'cp \"data/0_raw/{file}\" data/1_renamed/{str(i)}.xls'
            print_yellow(command)
            system(command)
    print_green(f'Finished renaming data in raw')

def preprocess_data_in_renamed():
    print_yellow(f'Started preprocessing data')
    from os import listdir
    from lib.submodules import preprocess_file
    dir = f'data/1_renamed'
    for file in listdir(dir):
        print_yellow(file)
        preprocess_file(dir, file)
    print_green(f'Finished preprocessing data')

def scale_features_in_preprocessed():
    from os import listdir, system
    dir = f'data/2_preprocessed'
    system(f'mkdir data/3_scaled')
    for file in listdir(dir):
        import pandas as pd
        df = pd.read_csv(f'{dir}/{file}')
        print(f'Length before:{df.shape[0]}')

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()

        scaler.fit(df)
        features_scaled = scaler.transform(df)
        import pandas as pd
        features_scaled_df = pd.DataFrame(features_scaled) 
        features_scaled_df.to_csv(f'data/3_scaled/{file}', index=False)
        print(f'Length After:{features_scaled_df.shape[0]}')

def window_and_score_files_in_scaled():
    print_yellow('Starting windowing')
    from os import listdir, system, path
    from lib.submodules import window_and_score_data
    if (not path.isdir('data/4_scored')):
        system('mkdir data/4_scored')
    dir = f'data/3_scaled'
    for file in listdir(dir):
        print_yellow(file)
        window_and_score_data(dir, file)
    print_green('Finished windowing')

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
def remap_names_lstm(model_name):
    """
    remap_names remaps names to original names

    @params
        filename : name of file
    """
    print_yellow("Starting Remap Names")
    import os
    mapping = open('data/mapping').read().splitlines()
    if not os.path.isdir('data/5_final_lstm'):
        os.system('mkdir data/5_final_lstm')
    animal=''
    if('mice' in model_name):
        animal='mouse'
    else:
        animal="rat"
    for i,file in enumerate(os.listdir('data/4_scored')):
        index_str = file.replace('.csv', '')
        newName = mapping[int(index_str)].replace('.xls', f'-lstm-{animal}.csv')
        os.system(f"cp data/4_scored/'{file}' data/5_final_lstm/'{newName}'")
    print_green("Finishing Remap Names")
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

## OLD zdb code
def zdb_preprocess():
    """
    zdb_preprocess preprocesses ZDBs for old ann/rf pipeline
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
    zdb_conversion imports csv into ZDB format. for old ann/rf pipeline
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
    zdb_remap remaps zdb names for old ann/rf pipeline
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

## New ZDB code for lstm pipeline
def rename_zdb_in_raw_zdb():
    """
    For lstm pipeline
    Rename zdb files in data/6_raw_zdb to data/7_renamed_zdb
    follows mapping of data files in data/mapping to rename files
    writes mapping of zdb files to data/mapping_zdb
    """
    raw_dir = '6_raw_zdb'
    renamed_dir = '7_renamed_zdb'
    os.system(f'mkdir -p data/{renamed_dir}')
    mapping = open('data/mapping').read().splitlines()
    mapping = [name.replace('.xls', '') for name in mapping]
    f = open('data/mapping_zdb','w+')
    i=0
    for name in mapping:
        for file in os.listdir(f"data/{raw_dir}"):
            if name in file or file.replace('.zdb', '') in name: #check for corrosponding names
                new_name = str(i)+'.zdb'
                os.system(f"cp data/{raw_dir}/{file}' data/{renamed_dir}/'{new_name}'")
                f.write(file+'\n')
                break
        i+=1
    f.close
    # use code in scripts/unzipAndRenameZDBData.py

def convert_zdb_in_renamed_zdb():
    """
    For lstm pipeline
    Uses scored csv's in data/5_final_lstm to add scores to zdb's 
        in data/7_renamed_zdb
    Saves new files in 8_scored_zdb
    """
    pass
    # copy files from 7_renamed_zdb to new dir 8_scored_zdb
    # pass 8_scored_zdb as dir_zdb
    # print_yellow("Starting ZDB Conversion")
    # import os
    # from lib.submodules import conversion_zdb

    # dir_zdb = 'data/renamed_zdb'
    # dir_scores = 'data/5_final_lstm'

    # if not os.path.isdir(dir_zdb):
    #     print("No ZDB files")
    #     return
    # for csv in os.listdir(dir_ann):
    #     name = csv.replace('.csv', '')
    #     zdb = f'{name}.zdb'
    #     conversion_zdb(dir_ann, dir_zdb, csv, zdb, 'ann')
    # for csv in os.listdir(dir_rf):
    #     name = csv.replace('.csv', '')
    #     zdb = f'{name}.zdb'
    #     conversion_zdb(dir_ann, dir_zdb, csv, zdb, 'rf')
    # print_green("Finishing ZDB Conversion")
def zdb_remap():
    """
    For lstm pipeline
    uses data/mapping_zdb to remap scored zdb files in data/8_renamed_zdb to 
        their original names in data/9_final_zdb_lstm
    """
    pass
    # print_yellow("Starting ZDB remap names")
    # import os
    # dir_ann = "data/ZDB_ann"
    # dir_rf = "data/ZDB_rf"
    # if not os.path.isdir(dir_ann) or not os.path.isdir(dir_rf):
    #     print("No ZDB files")
    #     return
    # os.system('mkdir data/ZDB_final_ann')
    # os.system('mkdir data/ZDB_final_rf')
    # mapping = open('data/ZDBmapping').read().splitlines()

    # for zdb in os.listdir(dir_ann):
    #     index = int(zdb.replace('.zdb', ''))
    #     os.system(f'cp {dir_ann}/"{zdb}" data/ZDB_final_ann/"{mapping[index]}"')
    # for zdb in os.listdir(dir_rf):
    #     index = int(zdb.replace('.zdb', ''))
    #     os.system(f'cp {dir_rf}/"{zdb}" data/ZDB_final_rf/"{mapping[index]}"')
    # print_green("Finished ZDB remapping")
def create_and_check_args():
    import argparse
    import os
    parser = argparse.ArgumentParser(description='Pipeline to Score Data')
    parser.add_argument('--ann-model', type=str, required=True, dest='ann_model', metavar='[path-to-model]',
                        help=f'Enter path to ANN model within {bcolors.BOLD}model{bcolors.ENDC} directory')
    args = parser.parse_args()
    if not os.path.exists('model/'+args.ann_model):
        print(f"{bcolors.FAIL}Invalid path to ANN model{bcolors.ENDC}")
        exit(1)
    return args