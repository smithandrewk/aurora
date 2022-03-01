#! /usr/bin/env python3

"""
Pipeline for ...
"""
print(__doc__)
from scripts.modules import *

process_timestamps()
initial_preprocessing()
handle_anomalies()
window()
scale()
score_ann()
score_rf()
expand_predictions()
remap_names()
zdb_conversion()