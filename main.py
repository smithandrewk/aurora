#!/usr/bin/env python3
from scripts.modules import *
import os

args = create_and_check_args()

create_time_dir()

## Run these only when we have new data
if args.new_data:
    rename_data_in_raw()
    preprocess_renamed_files()
    fix_anomalies_in_preprocessed_files()
    window_preprocessed_files()
    balance_windowed_files()
    concatenate_balanced_files()

## Run these to train new model
hln,baseline_history = load_data_and_train_model(args.data_dir)   
baseline_results = load_data_and_test_model(hln,baseline_history, args.data_dir)

