def initial_preprocessing():
    """
    initial_preprocessing does.

    @params
        filename : name of file
    """
    print("Starting Preprocessing")
    import os
    from scripts.submodules import preprocess_csv

    i = 0
    dir = f'data/renamed'
    for file in os.listdir(dir):
        print("Iteration " + str(i))
        print(dir,file)
        preprocess_csv(dir,file)
        i += 1
    print("Finishing Preprocessing")
def handle_anomalies():
    """
    handle_anomalies handles anomalies.

    @params
        filename : name of file
    """
    print("Starting Handling Anomalies")
    import pandas as pd
    from os import listdir
    # unscored_list = []
    for file in listdir("data/preprocessed"):
        df = pd.read_csv("data/preprocessed/"+file)
        print("======================================"+file)
        # if(df.columns[0]!="Class"):
        #     print("NOT SCORED")
        #     unscored_list.append("data/preprocessed/"+file)
        #     continue
        if(df.columns[1]!="0-0.5"):
            print("ANOMALY")
            df.rename(columns={"EEG 1 (0-0.5 Hz, 0-0.5Hz , 10s) (Mean, 10s)":"0-0.5"},inplace=True)
        EEG_2 = df["EEG 2"]
        Activity = df["Activity"]
        X = df.iloc[:,:-2]
        X.insert(X.shape[1],"EEG 2",EEG_2)
        X.insert(X.shape[1],"Activity",Activity)
        X.to_csv("data/preprocessed/"+file,index=False)

    print("Finishing Handling Anomalies")
def window():
    """
    window windows.

    @params
        filename : name of file
    """
    print("Starting Windowing")
    from scripts.submodules import window_data
    from os import listdir
    i = 0
    dir = f'data/preprocessed'
    for file in listdir(dir):
        print("Iteration: " + str(i))
        print(file)
        # X = read_csv(i)
        window_data(dir,file)
        i += 1
    print("Finishing Windowing")
def scale():
    """
    scale scales.

    @params
        filename : name of file
    """
    print("Starting Scaling")
    import pandas as pd
    import os
    i = 0
    dir = 'data/windowed'
    for file in os.listdir(dir):
        filename = f'{dir}/{file}'
        X = pd.read_csv(filename)
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        print("Iteration: " + str(i))
        X = scaler.fit_transform(X)
        if ( not os.path.isdir('data/windowed_scaled')):
            os.system('mkdir data/windowed_scaled')
        file = file.replace("_preprocessed_windowed.csv", "")
        filename = "data/windowed_scaled/"+file+"_windowed_scaled.csv"
        pd.DataFrame(X).to_csv(filename, index=False)
        print(filename)
        i += 1
    print("Finishing Scaling")
def score_ann(model):
    """
    score_ann scores ANN.

    @params
        model : name of model
    """
    print("Started Scoring ANN")
    from scripts.submodules import score_data_ann
    from os import listdir

    dir = 'data/windowed_scaled'
    i=0
    for file in listdir(dir):
        print("Iteration: " + str(i))
        score_data_ann(dir, file, model)
        i += 1

    print("Finishing Scoring ANN")
def score_rf(model):
    """
    score_rf scores RF.

    @params
        model : name of model
    """
    print("Starting Scoring RF")
    from scripts.submodules import score_data_rf
    from os import listdir

    dir = 'data/windowed'
    i=0
    for file in listdir(dir):
        print("Iteration: " + str(i))
        score_data_rf(dir, file, model)
        i+=1

    print("Finishing Scoring RF")
def expand_predictions():
    """
    expand_prediction expands predictions.

    @params
        filename : name of file
    """
    print("Starting Expand Predictions")
    from scripts.submodules import expand_predictions_ann, expand_predictions_rf
    from os import listdir

    dir_ann = 'data/predictions_ann'
    i=0
    print("Expanding ANN predictions")
    for file in listdir(dir_ann):
        print("Iteration: " + str(i))
        expand_predictions_ann(dir_ann, file)
        i+=1
    dir_rf = 'data/predictions_rf'
    print("Expanding RF predictions")
    i=0
    for file in listdir(dir_rf):
        print("Iteration: " + str(i))
        expand_predictions_rf(dir_rf, file)
        i+=1

    print("Finishing Expand Predictions")
def rename_scores():
    """
    rename_scores renames scores (0,1,2 --> P,S,W)
    """
    print("Starting Rename Scores")
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
    
    print("Finishing Rename Scores")
def remap_names():
    """
    remap_names remaps names to original names

    @params
        filename : name of file
    """
    print("Starting Remap Names")
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
    print("Finishing Remap Names")
def zdb_preprocess():
    """
    zdb_preprocess preprocesses ZDBs
    """
    print("Starting ZDB preprocessing")
    import os
    from scripts.submodules import preprocess_zdb
    dir = 'data/renamedZDB'
    if not os.isdir(dir):
        print("No ZDB files")
        return
    for file in os.listdir(dir):
        preprocess_zdb(dir, file)
    print("Finishing ZDB preprocessing")

def zdb_conversion():
    """
    zdb_conversion imports csv into ZDB format.
    """
    print("Starting ZDB Conversion")
    import os
    from scripts.submodules import ZDBconversion

    dir_zdb = 'data/preprocessedZDB'
    dir_ann = 'data/expanded_renamed_ann'
    dir_rf = 'data/expanded_renamed_rf'

    if not os.isdir(dir_zdb):
        print("No ZDB files")
        return
    for csv in os.listdir(dir_ann):
        name = csv.replace('.csv', '')
        zdb = f'{name}.zdb'
        ZDBconversion(dir_ann, dir_zdb, csv, zdb, 'ann')
    for csv in os.listdir(dir_rf):
        name = csv.replace('.csv', '')
        zdb = f'{name}.zdb'
        ZDBconversion(dir_ann, dir_zdb, csv, zdb, 'rf')
    print("Finishing ZDB Conversion")