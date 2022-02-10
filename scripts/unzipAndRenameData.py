#!/usr/bin/env python3
import os

os.system('mkdir data')
os.system('cp RawSleepData.tar.gz data/RawSleepData.tar.gz')
os.system('mkdir data/raw')
os.system('mkdir data/renamed')
os.system('gunzip data/RawSleepData.tar.gz')
os.system('tar -xvf data/RawSleepData.tar -C data/raw')

i = 0
f = open('data/mapping','w+')
for file in os.listdir("data/raw"):
	original_name = file
	file = file.replace(" ", "\ ")
	new_name = str(i)  + ".csv"
	cmd = "ssconvert data/raw/" + file + " data/renamed/" + new_name
	os.system(cmd)
	print("Iteration " + str(i) + ": Converting: " + str(original_name))
	f.write(original_name+'\n')
	i += 1
f.close()
