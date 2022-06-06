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