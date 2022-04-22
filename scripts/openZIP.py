#!/usr/bin/env python3

import os

os.system('mkdir -p data/raw')
os.system('cp Unscored.zip data/Unscored.zip')
os.system('unzip -j data/Unscored.zip -d ./data/raw')