#!/usr/bin/env python3
from scripts.modules import *
# rename_data_in_raw()
# preprocess_renamed_files()
# fix_anomalies_in_preprocessed_files()
# def select_features():
#     from os import listdir
#     from pandas import read_csv,DataFrame
#     dir = f'data/preprocessed'
#     for file in listdir(dir):
#         print(dir,file)
#         df = read_csv(f'{dir}/{file}')
#         Y = DataFrame()
#         if(df.columns[0]!="Class"):
#             print(f'Not scored, skipping')
#             continue
#         # Remove Unwanted Features
#         # df = df[['Class','EEG 2']]
#         # df = df[['Class','EEG 2','Activity']]
#         df = df.drop(columns=['Activity'])
#         df.to_csv(f'{dir}/{file}',index=False)
# select_features()
window_preprocessed_files()
balance_windowed_files()
concatenate_balanced_files()
split_and_shuffle('X.csv')
hln,baseline_history = load_data_and_train_model()
baseline_results = load_data_and_test_model(hln,baseline_history)
