#!/usr/bin/env python3
from scripts.modules import *
import os

args = create_and_check_args()
create_time_dir()

## Run these only when we have new data
# if args.new_data:
#     rename_data_in_raw()
#     preprocess_renamed_files()
#     fix_anomalies_in_preprocessed_files()
#     if args.skip_features:
#         skip_features(args.skip_features)
#     elif args.select_features:
#         select_features(args.select_features)
#     window_preprocessed_files()
#     balance_windowed_files()
#     concatenate_balanced_files()

if args.do_split_shuffle:
    args.data_dir = split_and_shuffle(args.data_dir)     #call split shuffle with data/X.csv (latest)
    print(args.data_dir)

## Run these to train new model
if args.do_train:
    hln,baseline_history = load_data_and_train_model(args.data_dir)   
    baseline_results = load_data_and_test_model(hln, baseline_history, args.data_dir)

if not args.skip_upload:
    upload_data(args.rclone_dir)