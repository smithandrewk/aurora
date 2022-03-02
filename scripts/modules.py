def initial_preprocessing():
    """
    initial_preprocessing does.

    @params
        filename : name of file
    """
    print("Starting Preprocessing")
    import os
    import pandas as pd
    from scripts.submodules import preprocess_csv

    i = 0
    dir = f'data/renamed'
    for file in os.listdir(dir):
        print("Iteration " + str(i))

        # TODO: need more logic here. What if a file doesn't have a time stamp column? <- Resolved, in preprocess submodule
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
    from pandas import read_csv
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