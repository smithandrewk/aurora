#!/usr/bin/env python3
from scripts.modules import *
create_time_dir()

## Run these only when we have new data
rename_data_in_raw()
preprocess_renamed_files()
# fix_anomalies_in_preprocessed_files()
window_preprocessed_files()
balance_windowed_files()
concatenate_balanced_files()

## Run these to train new model
# Options:
    # 1) dont run split_and_shuffle, must provide directory to load_data_and_train_model('03.23.2022_10:54') and load_data_and_test_model- will use previous sessions X, train, test, and val .csv
    # 2) run split_and_shuffle without directory - will use current sessions X.csv
    # 3) run split_and_shuffle with directory - will use X.csv from previous session to create new train,test,val .csv - must run load_data_and_train_model() without a directory 
# for the future, make simpler by making split_and_shuffle(dir) to do option 3?
# hln,baseline_history = load_data_and_train_model('03.23.2022_10:54', split_shuffle = True)
split_and_shuffle()
hln,baseline_history = load_data_and_train_model()

baseline_results = load_data_and_test_model(hln)

#'03.23.2022_10:54'