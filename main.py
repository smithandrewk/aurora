#!/usr/bin/env python3
from scripts.modules import *
rename_data_in_raw()
preprocess_renamed_files()
fix_anomalies_in_preprocessed_files()
select_features()
window_preprocessed_files()
balance_windowed_files()
concatenate_balanced_files()
split_and_shuffle('X.csv')
hln,baseline_history = load_data_and_train_model()
baseline_results = load_data_and_test_model(hln,baseline_history)
