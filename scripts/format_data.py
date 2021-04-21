#!/usr/bin/env python3
import os

os.system('rm -rf data/*')
os.system('mkdir data/raw')
os.system('mkdir data/test')
os.system('unzip -o bkp/SleepDepNeuralNetwork.zip -d data/raw')
i=0
f = open('data/mapping','w+')
for file in os.listdir("data/raw"):
    command = "cp \"data/raw/"+file+"\" data/test/"+str(i)+".xls"
    f.write(file+'\n')
    i+=1
    print(command)
    os.system(command)
f.close()
os.system('tar -xvf bkp/PrincesAuroraBackup.tar -C data/raw')
os.system('cp data/raw/10secScoredDataPowerControl.xls data/control.xls')
os.system('cp data/raw/10secScoredDataPowerSleepDeprivation.xls data/deprivation.xls')