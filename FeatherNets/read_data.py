from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset
import math
import cv2
import torchvision
import torch
import pandas as pd

# CASIA-SURF training dataset and our private dataset
# depth_dir_train_file = os.getcwd() +'/data/2depth_train.txt'
# label_dir_train_file = os.getcwd() + '/data/2label_train.txt'

dir_name = "/home/leyan/DataSet/CASIA-CeFA/phase1"
# CASIA-SURF training dataset and our private dataset
depth_dir_train_file = os.getcwd() +'/data/depth_train.txt'
label_dir_train_file = os.getcwd() + '/data/label_train.txt'

# for IR train
# depth_dir_train_file = os.getcwd() +'/data/ir_final_train.txt'
# label_dir_train_file = os.getcwd() +'/data/label_ir_train.txt'



# CASIA-SURF Val data 
depth_dir_val_file = os.getcwd() +'/data/depth_val.txt'
label_dir_val_file = os.getcwd() +'/data/label_val.txt' #val-label 100%


# depth_dir_val_file = os.getcwd() +'/data/ir_val.txt'
# label_dir_val_file = os.getcwd() +'/data/label_val.txt' #val-label 100%

# # CASIA-SURF Test data 
depth_dir_test_file = os.getcwd() +'/data/depth_test.txt'
label_dir_test_file = os.getcwd() +'/data/label_test.txt'


# depth_dir_test_file = os.getcwd() +'/data/ir_test.txt'
# label_dir_test_file = os.getcwd() +'/data/label_test.txt'

frames_total = 1

class CASIA(Dataset):
    def __init__(self, transform=None, phase_train=True, data_dir=None,phase_test=False):

        self.phase_train = phase_train
        self.phase_test = phase_test
        self.transform = transform

        try:
            with open(depth_dir_train_file, 'r') as f:
                self.depth_dir_train = f.read().splitlines()
            with open(label_dir_train_file, 'r') as f:
                self.label_dir_train = f.read().splitlines()
                
            with open(depth_dir_val_file, 'r') as f:
                 self.depth_dir_val = f.read().splitlines()
            with open(label_dir_val_file, 'r') as f:
                self.label_dir_val = f.read().splitlines()
            if self.phase_test:
                with open(depth_dir_test_file, 'r') as f:
                    self.depth_dir_test = f.read().splitlines()
                with open(label_dir_test_file, 'r') as f:
                    self.label_dir_test = f.read().splitlines()
        except:
            print('can not open files, may be filelist is not exist')
            exit()

    def __len__(self):
        if self.phase_train:
            return len(self.depth_dir_train)
        else:
            if self.phase_test:
                return len(self.depth_dir_test)
            else:
                return len(self.depth_dir_val)

    def __getitem__(self, idx):
        if self.phase_train:
            depth_dir = self.depth_dir_train
            label_dir = self.label_dir_train
            label = int(label_dir[idx])
            label = np.array(label)
        else:
            if self.phase_test:
                depth_dir = self.depth_dir_test
                label_dir = self.label_dir_test
#                 label = int(label_dir[idx])
                label = np.random.randint(0,2,1)
                label = np.array(label)
            else:
                depth_dir = self.depth_dir_val
                label_dir = self.label_dir_val
                label = int(label_dir[idx])
                label = np.array(label)

        depth = Image.open(depth_dir[idx])
        depth = depth.convert('RGB')

        if self.transform:
            depth = self.transform(depth)
        if self.phase_train:
            return depth,label
        else:
            return depth,label,depth_dir[idx]


class Spoofing_train(Dataset):

    def __init__(self, info_list, root_dir,  transform=None):

        self.landmarks_frame = pd.read_csv(info_list, delimiter=' ', header=None)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    
    def __getitem__(self, idx):
        #print(self.landmarks_frame.iloc[idx, 0])
        videoname = str(self.landmarks_frame.iloc[idx, 0])
        image_path = os.path.join(self.root_dir, videoname)
        
        # videoname_ir = videoname[:18] + 'ir/' + videoname[-8:]
        # ir_path = os.path.join(self.root_dir, videoname_ir)
        
        # videoname_depth = videoname[:18] + 'depth/' + videoname[-8:]
        # depth_path = os.path.join(self.root_dir, videoname_depth)

        videoname_l = videoname.split("/")
        videoname_ir = os.path.join(videoname_l[0], videoname_l[1]) + '/ir/' + os.path.basename(videoname)
        ir_path = os.path.join(self.root_dir, videoname_ir)
        
        videoname_depth = os.path.join(videoname_l[0], videoname_l[1]) + '/depth/' + os.path.basename(videoname)
        depth_path = os.path.join(self.root_dir, videoname_depth)
    
    
             
        image_x, image_ir, image_depth = self.get_single_image_x(image_path, ir_path, depth_path)
        
        
            
        spoofing_label = self.landmarks_frame.iloc[idx, 1]
        if spoofing_label == 1:
            spoofing_label = 1            # real
        else:
            spoofing_label = 0            # fake
        
        
        #frequency_label = self.landmarks_frame.iloc[idx, 2:2+50].values  

        # sample = {'image_x': image_x, 'image_ir': image_ir, 'image_depth': image_depth, 'binary_mask': binary_mask, 'spoofing_label': spoofing_label}

        # if self.transform:
        #     sample = self.transform(sample)

        # sample['string_name']= videoname
        # return sample

        if self.transform:
            image_depth = self.transform(image_depth)
        if self.phase_train:
            return image_depth,spoofing_label
        else:
            return image_depth,spoofing_label,depth_path



    def get_single_image_x(self, image_path, ir_path, depth_path):
        
        
        image_x = np.zeros((256, 256, 3))
        binary_mask = np.zeros((32, 32))
 
 
        image_x_temp = cv2.imread(image_path)
        image_x_temp_ir = cv2.imread(ir_path)
        image_x_temp_depth = cv2.imread(depth_path)

        return image_x_temp, image_x_temp_ir, image_x_temp_depth
        # image_x_temp_gray = cv2.imread(image_path, 0)


        # image_x = cv2.resize(image_x_temp, (256, 256))
        # image_x_ir = cv2.resize(image_x_temp_ir, (256, 256))
        # image_x_depth = cv2.resize(image_x_temp_depth, (256, 256))
        # image_x_temp_gray = cv2.resize(image_x_temp_gray, (32, 32))
        # image_x_aug = seq.augment_image(image_x) 
        # image_x_aug_ir = seq.augment_image(image_x_ir) 
        # image_x_aug_depth = seq.augment_image(image_x_depth) 
        
             
        
        # for i in range(32):
        #     for j in range(32):
        #         if image_x_temp_gray[i,j]>0:
        #             binary_mask[i,j]=1
        #         else:
        #             binary_mask[i,j]=0
        
        
        
   
        # return image_x_aug, image_x_aug_ir, image_x_aug_depth, binary_mask


class Spoofing_valtest(Dataset):

    def __init__(self, info_list, root_dir,  transform=None):

        self.landmarks_frame = pd.read_csv(info_list, delimiter=' ', header=None)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    
    def __getitem__(self, idx):

        videoname = str(self.landmarks_frame.iloc[idx, 0]) # + "_" + str(self.landmarks_frame.iloc[idx, 1])
        try:
            live = float(self.landmarks_frame.iloc[idx, 1])
        except:
            live = float(0.0)
        image_path = os.path.join(self.root_dir, videoname)
        image_path2 = os.path.join(image_path, 'profile')
        ir_path = os.path.join(image_path, 'ir')
        depth_path = os.path.join(image_path, 'depth')
             
        image_x, image_ir, image_depth, binary_mask = self.get_single_image_x(image_path2, ir_path, depth_path, videoname)
                        
        sample = {'image_x': image_x,'image_ir': image_ir,'image_depth': image_depth, 'binary_mask': binary_mask, 'string_name': videoname}

        if self.transform:
            sample = self.transform(sample)
        sample['live'] = live
        sample["depth_dirs"] = depth_path
        return sample

    def get_single_image_x(self, image_path, ir_path, depth_path, videoname):

        files_total = len([name for name in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, name))])
        interval = files_total//frames_total
        
        image_x = np.zeros((frames_total, 256, 256, 3))
        image_ir = np.zeros((frames_total, 256, 256, 3))
        image_depth = np.zeros((frames_total, 256, 256, 3))
        
        binary_mask = np.zeros((frames_total, 32, 32))
        
        
        
        # random choose 1 frame
        for ii in range(frames_total):
            image_id = ii*interval + 1 
            
            s = "%04d.jpg" % image_id            
            
            # RGB
            image_path2 = os.path.join(image_path, s)
            image_x_temp = cv2.imread(image_path2)
            
            # ir
            image_path2_ir = os.path.join(ir_path, s)
            image_x_temp_ir = cv2.imread(image_path2_ir)
            
            # depth
            image_path2_depth = os.path.join(depth_path, s)
            image_x_temp_depth = cv2.imread(image_path2_depth)
            
            image_x_temp_gray = cv2.imread(image_path2, 0)
            image_x_temp_gray = cv2.resize(image_x_temp_gray, (32, 32))

            image_x[ii,:,:,:] = cv2.resize(image_x_temp, (256, 256))
            image_ir[ii,:,:,:] = cv2.resize(image_x_temp_ir, (256, 256))
            image_depth[ii,:,:,:] = cv2.resize(image_x_temp_depth, (256, 256))

            # image_x = image_x.astype(np.uint8)
            # image_ir = image_ir.astype(np.uint8)
            # image_depth = image_depth.astype(np.uint8)

            # cv2.imshow("1", cv2.resize(image_x_temp, (256, 256)))
            # cv2.imshow("2", image_x[ii,:,:,:])
            # cv2.imshow("3", image_ir[ii,:,:,:])
            # cv2.imshow("4", image_depth[ii,:,:,:])
            # cv2.waitKey(0)
            
            #print(image_path2)
            
            for i in range(32):
                for j in range(32):
                    if image_x_temp_gray[i,j]>0:
                        binary_mask[ii, i, j]=1.0
                    else:
                        binary_mask[ii, i, j]=0.0
            

        
        return image_x, image_ir, image_depth, binary_mask

