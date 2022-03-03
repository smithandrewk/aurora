#! /usr/bin/env python3

"""
Pipeline for ...
"""
print(__doc__)
from scripts.modules import *

initial_preprocessing()
handle_anomalies()
window()
scale()
score_ann('best_model.h5')
score_rf('rf_model')
expand_predictions()
rename_scores()
remap_names()
zdb_conversion()