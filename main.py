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
hln,baseline_history = load_data_and_train_model(dir)
baseline_results = load_data_and_test_model(hln, dir)

#'03.23.2022_10:54'