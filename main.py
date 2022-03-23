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
# To skip split and shuffle, provide directory to functions to read data (eg. "03.23.2022_10:54") from instead
hln,baseline_history = load_data_and_train_model('03.23.2022_10:54') 
baseline_results = load_data_and_test_model(hln, '03.23.2022_10:54')
