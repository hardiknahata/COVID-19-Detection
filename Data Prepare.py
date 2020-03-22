import os

path1 = "dataset/covid/"
path2 = "dataset/normal/"

i=0

for filename in os.listdir(path1):
    dst ="covid" + str(i) + ".jpg"
    src = path1 + filename 
    dst = path1 + dst
    os.rename(src, dst) 
    i += 1

i=0

for filename in os.listdir(path2):
    dst ="normal" + str(i) + ".jpg"
    src = path2 + filename 
    dst = path2 + dst
    os.rename(src, dst) 
    i += 1