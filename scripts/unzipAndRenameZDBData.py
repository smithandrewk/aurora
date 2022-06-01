#!/usr/bin/env python3

import os

os.system('mkdir data')
os.system('mkdir data/rawZDB')
os.system('mkdir data/renamedZDB')

os.system('cp UnscoredZDB.zip data/UnscoredZDB.zip')
os.system('unzip -j data/UnscoredZDB.zip -x / -d ./data/rawZDB')

mapping = open('data/mapping').read().splitlines()
mapping = [name.replace('.xls', '') for name in mapping]

f = open('data/ZDBmapping','w+')
i=0
for name in mapping:
    for file in os.listdir("data/rawZDB"):
        if name in file or file.replace('.zdb', '') in name: #check for corrosponding names
            newName = str(i)+'.zdb'
            os.system("cp data/rawZDB/'"+file+"' data/renamedZDB/'"+newName+"'")
            f.write(file+'\n')
            break
    i+=1
f.close