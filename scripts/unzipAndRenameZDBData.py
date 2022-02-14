#!/usr/bin/env python3

import os

os.system('mkdir data/rawZDB')
os.system('mkdir data/renamedZDB')

os.system('cp UnscoredZDB.zip data/UnscoredZDB.zip')
os.system('unzip -j data/UnscoredZDB.zip -x / -d ./data/rawZDB')

mapping = mapping = open('data/mapping').read().splitlines()
mapping = [name.replace('.xls', '') for name in mapping]
for file in os.listdir("data/rawZDB"):
    i=0
    newName = ''
    for name in mapping:
        if name in file:
            newName = str(i)+'.zdb'
        i+=1
    os.system("cp data/rawZDB/'"+file+"' data/renamedZDB/'"+newName+"'")


