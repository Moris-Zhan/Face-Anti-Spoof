
train_4_1 =  open("/home/leyan/DataSet/CASIA-CeFA/phase1/4@1_train.txt").readlines()    
train_4_2 =  open("/home/leyan/DataSet/CASIA-CeFA/phase1/4@2_train.txt").readlines()    
train_4_3 =  open("/home/leyan/DataSet/CASIA-CeFA/phase1/4@3_train.txt").readlines() 

train_set = []

train_set.extend( train_4_1)
train_set.extend( train_4_2)
train_set.extend( train_4_3)
# train_set = [line.strip() for line in train_set]
A = len(train_set)
B = len(set(train_set))
pass

with open("/home/leyan/DataSet/CASIA-CeFA/phase1/train.txt", "w") as f:
    f.writelines(train_set)