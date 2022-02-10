#!/usr/bin/env python3
import os

os.system('mkdir data')
os.system('cp Unscored-Data-Snezana-11-12-21.zip data/Unscored-Data-Snezana-11-12-21.zip')
os.system('mkdir data/raw')
os.system('mkdir data/renamed')
#os.system('gunzip data/Unscored-Data-Snezana-11-12-21')
os.system('unzip data/Unscored-Data-Snezana-11-12-21.zip -d data/raw ')

i = 0
f = open('data/mapping','w+')
for file in os.listdir("data/raw"):
	original_name = file
	file = file.replace(" ", "\ ")
	original_name = original_name.replace(" ", "\ ")
	original_name = original_name.replace(".xls", ".csv")
	new_name = str(i)  + ".csv"
	# cmd = "ssconvert data/raw/" + file + " data/renamed/" + new_name
	cmd = "ssconvert data/raw/" + file + " data/renamed/" + original_name
	os.system(cmd)
	print("Iteration " + str(i) + ": Converting: " + str(original_name))
	f.write(original_name+'\n')
	i += 1
f.close()
