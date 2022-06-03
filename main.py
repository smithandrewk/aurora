#! /usr/bin/env python3

"""
Pipeline for ...
"""
print(__doc__)
from lib.modules import *

# args = create_and_check_args()

# rename_data_in_raw()
# preprocess_data_in_renamed()
# scale_features_in_preprocessed()
# window_and_score_files_in_scaled()
# remap_names_lstm(args.ann_model)

rename_files_in_raw_zdb()
score_files_in_renamed_zdb()
remap_files_in_scored_zdb()

# score_ann(args.ann_model)
# score_rf('rf_model')
# expand_predictions()
# rename_scores()
# remap_names()
# zdb_preprocess()
# zdb_conversion()
# zdb_remap()