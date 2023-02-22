from glob import glob
import os
import pandas as pd

dir = os.path.join(os.getcwd(), "phase2")

def Sequence_count(protocol="4@1"):    
    dir= "/home/leyan/DataSet/CASIA-CeFA/phase2/"
    file_list = f"/home/leyan/DataSet/CASIA-CeFA/phase2/{protocol}_test_res.txt"
    lines = open(file_list).readlines()
    lines = [os.path.join(dir, line.strip(),"profile/*.jpg")  for line in lines]
    dir_count = [len(glob(line)) for line in lines]
    return dir_count


# data = {'4@1': Sequence_count("4@1"),
#         '4@2': Sequence_count("4@2"),
#         '4@3': Sequence_count("4@3"),
#         }

# df = pd.DataFrame(data)
# print("Test Protocol: ", df.describe())
# df_desc = df.describe()
# df_desc.to_csv("Test-Protocol.csv")

####################################################################################
def Sequence_count_dev(protocol="4@1"):    
    dir= "/home/leyan/DataSet/CASIA-CeFA/phase1/"
    file_list = f"/home/leyan/DataSet/CASIA-CeFA/phase1/{protocol}_dev_ref.txt"
    lines = open(file_list).readlines()
    labels = [int(line.strip().split(" ")[1])  for line in lines]
    lines = [os.path.join(dir, line.strip().split(" ")[0],"profile/*.jpg")  for line in lines]
    dir_count = [len(glob(line)) for line in lines]
    print(protocol, " Living counts: ", sum(labels))
    print(protocol, " Spoof counts: ", len(labels) - sum(labels))
    return dir_count


data = {'4@1': Sequence_count_dev("4@1"),
        '4@2': Sequence_count_dev("4@2"),
        '4@3': Sequence_count_dev("4@3"),
        }

df = pd.DataFrame(data)
print("Valid Protocol: ", df.describe())
df_desc = df.describe()
df_desc.to_csv("Valid-Protocol.csv")

####################################################################################
def Sequence_count_dev(protocol="4@1"):    
    dir= "/home/leyan/DataSet/CASIA-CeFA/phase1/"
    file_list = f"/home/leyan/DataSet/CASIA-CeFA/phase1/{protocol}_train.txt"
    lines = open(file_list).readlines()
    # lines = [os.path.join(dir, line.strip().split(" ")[0])  for line in lines]
    labels = [int(line.strip().split(" ")[1])  for line in lines]
    # dir_count = [len(glob(line)) for line in lines]
    print(protocol, " Living counts: ", sum(labels))
    print(protocol, " Spoof counts: ", len(labels) - sum(labels))
    return labels



data = {'4@1': Sequence_count_dev("4@1"),
        '4@2': Sequence_count_dev("4@2"),
        '4@3': Sequence_count_dev("4@3"),
        }
pass