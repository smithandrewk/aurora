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
# Pass directory name (eg. '03.24.2022_12:40') to load_data_and_train_model() and load_data_and_test_model() to use data from a previous session
hln,baseline_history = load_data_and_train_model('03.24.2022_12:40')   
baseline_results = load_data_and_test_model(hln,baseline_history,'03.24.2022_12:40')

