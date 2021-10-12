#!/usr/bin/env python3
import os

os.system('rm -rf data/*')
os.system('mkdir data/raw')
os.system('mkdir data/renamed')
os.system('unzip -o bkp/SleepDepNeuralNetwork.zip -d data/raw')
os.system('tar -xvf bkp/PrincesAuroraBackup.tar -C data/raw')

i=0
f = open('data/mapping','w+')
for file in os.listdir("data/raw"):
    command = "cp \"data/raw/"+file+"\" data/renamed/"+str(i)+".xls"
    f.write(file+'\n')
    i+=1
    print(i)
    print(command)
    os.system(command)
f.close()