
dev_ref_4_1 =  open("/home/leyan/DataSet/CASIA-CeFA/phase1/4@1_dev_ref.txt").readlines()    
dev_ref_4_2 =  open("/home/leyan/DataSet/CASIA-CeFA/phase1/4@2_dev_ref.txt").readlines()    
dev_ref_4_3 =  open("/home/leyan/DataSet/CASIA-CeFA/phase1/4@3_dev_ref.txt").readlines() 

dev_ref_set = []

dev_ref_set.extend( dev_ref_4_1)
dev_ref_set.extend( dev_ref_4_2)
dev_ref_set.extend( dev_ref_4_3)
# dev_ref_set = [line.strip() for line in dev_ref_set]
A = len(dev_ref_set)
B = len(set(dev_ref_set))

with open("/home/leyan/DataSet/CASIA-CeFA/phase1/dev.txt", "w") as f:
    f.writelines(dev_ref_set)