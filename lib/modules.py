from .utils import print_on_start_on_end,bcolors,print_yellow,execute_command_line
@print_on_start_on_end
def create_and_check_args():
    import argparse
    import os
    parser = argparse.ArgumentParser(description='Pipeline to Score Data')
    parser.add_argument('--ann-model', type=str, required=True, dest='ann_model', metavar='[path-to-model]',
                        help=f'Enter relative or absolute path to your model')
    args = parser.parse_args()

    if not os.path.exists(args.ann_model):
        print(f"{bcolors.FAIL}Invalid path to ANN model{bcolors.ENDC}")
        exit(1)
    return args

@print_on_start_on_end
def rename_data_in_raw():
    from os.path import isdir
    from os import listdir, system

    if not isdir('data'):
        print(f"{bcolors.FAIL}'/data' directory does not exist. create data directory and place your raw data into a subdirectory called '0_raw'{bcolors.ENDC}")
        exit(1)

    if not isdir('data/0_raw'):
        print(f"{bcolors.FAIL}'/data/0_raw' directory does not exist. create a subdirectory in '/data' called '0_raw' and place your raw unscored files in there{bcolors.ENDC}")
        exit(1)

    if isdir('data/1_renamed'):
        print(f'/data/1_renamed exists. deleting and recreating.')
        execute_command_line(f'rm -rf data/1_renamed')
    execute_command_line(f'mkdir data/1_renamed')

    with open('data/mapping', 'w+') as f:
        src_dir = 'data/0_raw'
        dest_dir = 'data/1_renamed'
        # TODO : add header, but don't want to right now because it will break things downstream
        for i, file in enumerate(listdir(src_dir)):
            f.write(f'{i},{file}\n')
            execute_command_line(f'cp \"{src_dir}/{file}\" {dest_dir}/{str(i)}.xls')

@print_on_start_on_end
def preprocess_data_in_renamed():
    from os import listdir
    from lib.submodules import preprocess_file
    dir = f'data/1_renamed'
    for file in listdir(dir):
        preprocess_file(dir, file)

@print_on_start_on_end
def scale_features_in_preprocessed():
    from os import listdir
    from sklearn.preprocessing import StandardScaler
    import pandas as pd

    dir = f'data/2_preprocessed'
    execute_command_line(f'mkdir data/3_scaled')
    for file in listdir(dir):
        import pandas as pd
        df = pd.read_csv(f'{dir}/{file}')
        len_before = df.shape[0]
        scaler = StandardScaler()
        scaler.fit(df)
        df = pd.DataFrame(scaler.transform(df))
        df.to_csv(f'data/3_scaled/{file}', index=False)
        print(f'Length Before: {len_before}\nLength After : {df.shape[0]}')

@print_on_start_on_end
def window_and_score_files_in_scaled_with_LSTM(path_to_model):
    from os import listdir
    from os.path import isdir
    from lib.submodules import window_and_score_data
    from keras.models import load_model
    if not isdir('data/4_scored'):
        execute_command_line('mkdir data/4_scored')
    model = load_model(path_to_model)
    dir = f'data/3_scaled'
    for file in listdir(dir):
        window_and_score_data(dir,file,model)

@print_on_start_on_end
def remap_names_lstm(model_name):
    """
    remap_names remaps names to original names

    @params
        filename : name of file
    """
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
        line = mapping[int(index_str)]
        original_name = line.split(',')[1]
        newName = original_name.replace('.xls', f'-lstm-{animal}.csv')
        execute_command_line(f"cp data/4_scored/'{file}' data/5_final_lstm/'{newName}'")

@print_on_start_on_end
def rename_files_in_raw_zdb():
    """
    For lstm pipeline
    Rename zdb files in data/6_raw_zdb to data/7_renamed_zdb
    follows mapping of data files in data/mapping to rename files
    writes mapping of zdb files to data/mapping_zdb
    """
    import os
    old_dir = 'data/6_raw_zdb'
    new_dir = 'data/7_renamed_zdb'
    
    if not os.path.isdir(old_dir):
        print("No ZDB files")
        return
    os.system(f'mkdir -p {new_dir}')
    mapping = open('data/mapping').read().splitlines()
    mapping = [name.replace('.xls', '') for name in mapping]
    num_files = len(mapping)
    f = open('data/mapping_zdb','w+')
    i=0
    for line in mapping:
        name = line.split(',')[1]
        for file in os.listdir(f"{old_dir}"):
            if name in file or file.replace('.zdb', '') in name: #check for corrosponding names
                new_name = str(i)+'.zdb'
                os.system(f"cp {old_dir}/'{file}' {new_dir}/'{new_name}'")
                f.write(file+'\n')
                i+=1
                break
    f.close
    if i != num_files:
        print('ERROR: XLS file names do not corrospong to ZDB file names')
        os.system(f'rm -rf {new_dir}')

@print_on_start_on_end
def score_files_in_renamed_zdb():
    """
    For lstm pipeline
    Uses scored csv's in data/5_final_lstm to add scores to zdb's 
        in data/7_renamed_zdb
    Saves new files in 8_scored_zdb
    """
    from lib.submodules import convert_zdb_lstm
    import os

    old_dir = 'data/7_renamed_zdb'
    new_dir = 'data/8_scored_zdb'
    csv_dir = 'data/4_scored'

    if not os.path.isdir(old_dir):
        print("No ZDB files")
        return
    
    os.system(f'mkdir -p {new_dir}')

    for file in os.listdir(old_dir):
        os.system(f"cp {old_dir}/{file} {new_dir}/{file}")

    for csv in os.listdir(csv_dir):
        zdb = csv.replace('.csv', '.zdb')
        convert_zdb_lstm(csv_dir, new_dir, csv, zdb)

@print_on_start_on_end
def remap_files_in_scored_zdb(model):
    """
    For lstm pipeline
    uses data/mapping_zdb to remap scored zdb files in data/8_renamed_zdb to 
        their original names in data/9_final_zdb_lstm
    """
    import os
    old_dir = 'data/8_scored_zdb'
    new_dir = 'data/9_final_zdb_lstm'

    if not os.path.isdir(old_dir):
        print("No ZDB files")
        return
    os.system(f'mkdir -p {new_dir}')
    mapping = open('data/mapping_zdb').read().splitlines()

    animal=''
    if('mice' in model):
        animal='mouse'
    else:
        animal="rat"

    for zdb in os.listdir(old_dir):
        index = int(zdb.replace('.zdb', ''))
        new_name = mapping[index].replace('.zdb', f'-lstm-{animal}.zdb')
        os.system(f"cp {old_dir}/'{zdb}' {new_dir}/'{new_name}'")
def create_and_check_args():
    import argparse
    import os
    parser = argparse.ArgumentParser(description='Pipeline to Score Data')
    parser.add_argument('--ann-model', type=str, required=True, dest='ann_model', metavar='[path-to-model]',
                        help=f'Enter path to ANN model within {bcolors.BOLD}model{bcolors.ENDC} directory')
    args = parser.parse_args()
    if not os.path.exists('model/'+args.ann_model) and not os.path.exists(args.ann_model):
        print(f"{bcolors.FAIL}Invalid path to ANN model{bcolors.ENDC}")
        exit(1)
    return args
