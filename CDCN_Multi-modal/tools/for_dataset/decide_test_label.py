import cv2
from glob import glob
import os
import numpy as np

lines = []
with open("/home/leyan/DataSet/CASIA-CeFA/phase2/test.txt", "r") as f:
    lines = f.readlines()

    lines = [line.strip() for line in lines]


skip = len(open("/home/leyan/DataSet/CASIA-CeFA/phase2/test_res.txt", "r").readlines())
w = open("/home/leyan/DataSet/CASIA-CeFA/phase2/test_res.txt", "a")
for idx , name in enumerate(lines):
    print(f"label {name}")
    if idx < skip: 
        continue

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
    if c == 81:
        label = False
    elif c == 83:
        label = True
    if c==27:
        break
    w.write(f"{name} {int(label)} \n")
    print(name, f"label live {label}")
    cv2.destroyAllWindows()

    # /home/leyan/DataSet/CASIA-CeFA/phase2/test/001685/depth/0001.jpg
w.close()