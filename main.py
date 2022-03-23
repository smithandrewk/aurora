#!/usr/bin/env python3
from scripts.modules import *
create_time_dir()

## Run these only when we have new data
# rename_data_in_raw()
# preprocess_renamed_files()
# fix_anomalies_in_preprocessed_files()
# window_preprocessed_files()
# balance_windowed_files()
# concatenate_balanced_files()

## Run these to train new model
# Options:
    # 1) give nothing to load_data_and_train_model  ->  read all training data from this session's data
    # 2) give directory (eg. "03.23.2022_10:54") to load_data_and_train_model  ->  read all training data from that directory
    # 3) give directory to load_data_and_train_model and set split_shuffle = True ->  read X.csv from that directory and split and shuffle with that X.csv
    # for Option 2 only, give directory to load_data_and_test_model as well
# for the future, make simpler by making split_and_shuffle(dir) to do option 3?
# hln,baseline_history = load_data_and_train_model('03.23.2022_10:54', split_shuffle = True)
hln,baseline_history = load_data_and_train_model('03.23.2022_10:54', split_shuffle = True)

baseline_results = load_data_and_test_model(hln, '03.23.2022_10:54')