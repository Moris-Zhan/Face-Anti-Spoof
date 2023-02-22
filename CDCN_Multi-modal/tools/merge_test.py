
test_res_4_1 =  open("/home/leyan/DataSet/CASIA-CeFA/phase2/4@1_test_res.txt").readlines()    
test_res_4_2 =  open("/home/leyan/DataSet/CASIA-CeFA/phase2/4@2_test_res.txt").readlines()    
test_res_4_3 =  open("/home/leyan/DataSet/CASIA-CeFA/phase2/4@3_test_res.txt").readlines() 

test_res_set = []

test_res_set.extend( test_res_4_1)
test_res_set.extend( test_res_4_2)
test_res_set.extend( test_res_4_3)
# test_res_set = [line.strip() for line in test_res_set]
A = len(test_res_set)
B = len(set(test_res_set))

with open("/home/leyan/DataSet/CASIA-CeFA/phase2/test.txt", "w") as f:
    f.writelines(list(set(test_res_set)))