#!/usr/bin/env python3
from scripts.modules import *
time = get_time_dir()
rename_data_in_raw()
preprocess_renamed_files()
fix_anomalies_in_preprocessed_files()
window_preprocessed_files()
balance_windowed_files()
concatenate_balanced_files(time)
split_and_shuffle(f'X.csv', time)
hln,baseline_history = load_data_and_train_model(time)
baseline_results = load_data_and_test_model(hln)
