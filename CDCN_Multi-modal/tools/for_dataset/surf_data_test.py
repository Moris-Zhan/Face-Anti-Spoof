import os
from glob import glob

directory = "/home/leyan/DataSet/Surfing-2020-Anti-spoofing/data"

import os,string
path = directory
path = os.path.normpath(path)
res = []
for root,dirs,files in os.walk(path, topdown=True):
    depth = root[len(path) + len(os.path.sep):].count(os.path.sep)
    if depth == 2:
        # We're currently two directories in, so all subdirs have depth 3
        res += [os.path.join(root, d) for d in dirs]
        dirs[:] = [] # Don't recurse any deeper

res.sort()

for dir in res:
    print(dir)

print("Body: ", len([r for r in res if "body" in r]))
print("3DMask: ", len([r for r in res if "3DMask" in r]))
print("Paper-Attack: ", len([r for r in res if "Paper-Attack" in r]))


"""make surf_ref.txt"""
fn = "/home/leyan/DataSet/Surfing-2020-Anti-spoofing/data/surf_ref.txt"
with open(fn, "w") as f:
    for dir in res:
        label = 1
        if "real" not in dir: 
            label = 0
        f.write(f"{dir} {label}\n")