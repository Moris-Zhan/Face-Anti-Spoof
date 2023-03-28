import os
import cv2
from glob import glob
import numpy as np


ref_41 = open("/home/leyan/DataSet/CASIA-CeFA/phase2/4@1_test_ref.txt", "r").readlines() 
ref_42 = open("/home/leyan/DataSet/CASIA-CeFA/phase2/4@2_test_ref.txt", "r").readlines() 

ref_41 = [(os.path.join(line.strip().split(" ")[0],"profile/0001.jpg"), line.strip().split(" ")[1]) for line in ref_41]
ref_42 = [(line.strip().split(" ")[0], line.strip().split(" ")[1]) for line in ref_42]

b_train = open("/home/leyan/DataSet/CASIA-CeFA/phase1/b_train.txt", "a")
b_dev= open("/home/leyan/DataSet/CASIA-CeFA/phase1/b_dev.txt", "a")

for idx , (name,label) in enumerate(ref_41): 
    b_train.write(f"{name} {int(label)}\n")

for idx , (name,label) in enumerate(ref_42): 
    b_dev.write(f"{name} {int(label)}\n")

pass