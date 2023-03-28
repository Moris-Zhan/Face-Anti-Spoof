import os
import cv2
from glob import glob
import numpy as np


lines = open("/home/leyan/DataSet/CASIA-CeFA/phase2/test_res.txt", "r").readlines()
for idx , name in enumerate(lines):
    label = int(name.split(" ")[1])
    name = name.split(" ")[0]
    if idx < 500: continue
    if label != 1: continue 
    print(f"label {name}")    

    depth = cv2.imread(glob(os.path.join("/home/leyan/DataSet/CASIA-CeFA/phase2", name, "depth/0001.jpg"))[0])
    depth = cv2.resize(depth, (512, 512))

    ir = cv2.imread(glob(os.path.join("/home/leyan/DataSet/CASIA-CeFA/phase2", name, "ir/0001.jpg"))[0])
    ir = cv2.resize(ir, (512, 512))

    rgb = cv2.imread(glob(os.path.join("/home/leyan/DataSet/CASIA-CeFA/phase2", name, "profile/0001.jpg"))[0])
    rgb = cv2.resize(rgb, (512, 512))
    img = np.hstack([rgb, ir, depth, ])
    # RGB+Depth
    cv2.imshow(name, img)
    c= cv2.waitKey(0)

    live = True    
    if c==27:
        break
    print(name, f"label live {label}")
    cv2.destroyAllWindows()