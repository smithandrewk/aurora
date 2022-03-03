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
def score_ann():
    """
    scale scales.

    @params
        filename : name of file
    """
    print("Starting Scaling")

    print("Finishing Scaling")
def score_rf():
    """
    scale scales.

    @params
        filename : name of file
    """
    print("Starting Scaling")

    print("Finishing Scaling")
def expand_predictions():
    """
    scale scales.

    @params
        filename : name of file
    """
    print("Starting Scaling")

    print("Finishing Scaling")
def remap_names():
    """
    scale scales.

    @params
        filename : name of file
    """
    print("Starting Scaling")

    print("Finishing Scaling")
def zdb_conversion():
    """
    scale scales.

    @params
        filename : name of file
    """
    print("Starting Scaling")

    print("Finishing Scaling")