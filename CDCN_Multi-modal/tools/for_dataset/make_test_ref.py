import os
import cv2
from glob import glob
import numpy as np

lines = open("/home/leyan/DataSet/CASIA-CeFA/phase2/test_res.txt", "r").readlines()

res_41 = open("/home/leyan/DataSet/CASIA-CeFA/phase2/4@1_test_res.txt", "r").readlines() 
res_42 = open("/home/leyan/DataSet/CASIA-CeFA/phase2/4@2_test_res.txt", "r").readlines() 
res_43 = open("/home/leyan/DataSet/CASIA-CeFA/phase2/4@3_test_res.txt", "r").readlines()  

res_41 = {line.strip() : 0 for line in res_41}
res_42 = {line.strip() : 0 for line in res_42}
res_43 = {line.strip() : 0 for line in res_43}
lines = [line.strip() for line in lines]

for line in lines:
    label = int(line.split(" ")[1])
    name = line.split(" ")[0]
    if name in res_41.keys():  
        res_41[name] = label
        print("write into 4@1 ", name, label)
    if name in res_42.keys():  
        res_42[name] = label
        print("write into 4@2 ", name, label)
    if name in res_43.keys():  
        res_43[name] = label
        print("write into 4@3 ", name, label)
    


ref_41 = open("/home/leyan/DataSet/CASIA-CeFA/phase2/4@1_test_ref.txt", "w")
ref_42 = open("/home/leyan/DataSet/CASIA-CeFA/phase2/4@2_test_ref.txt", "w")
ref_43 = open("/home/leyan/DataSet/CASIA-CeFA/phase2/4@3_test_ref.txt", "w")

for idx , (name,label) in enumerate(res_41.items()):    
    ref_41.write(f"{name} {int(label)} \n")
ref_41.close()

for idx , (name,label) in enumerate(res_42.items()):    
    ref_42.write(f"{name} {int(label)} \n")
ref_42.close()

for idx , (name,label) in enumerate(res_43.items()):    
    ref_43.write(f"{name} {int(label)} \n")
ref_43.close()