#! /usr/bin/env python3

"""
Pipeline for ...
"""
print(__doc__)
from scripts.modules import *

# initial_preprocessing()
# handle_anomalies()
# window()
# scale()
score_ann('mice_512hln_ann_96.4_accuracy/best_model.h5')
score_rf('rf_model')
expand_predictions()
rename_scores()
remap_names()
zdb_preprocess()
zdb_conversion()
zdb_remap()