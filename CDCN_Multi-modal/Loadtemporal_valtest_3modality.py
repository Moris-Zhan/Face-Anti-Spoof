from __future__ import print_function, division
import os
import torch
import pandas as pd
#from skimage import io, transform
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pdb
import math
import os 
from glob import glob


frames_total = 8    # each video 8 uniform samples


class Normaliztion_valtest(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """
    def __call__(self, sample):
        image_x, image_ir, image_depth, binary_mask, string_name = sample['image_x'], sample['image_ir'], sample['image_depth'],sample['binary_mask'],sample['string_name']
        
        new_image_x = (image_x - 127.5)/128     # [-1,1]
        new_image_ir = (image_ir - 127.5)/128     # [-1,1]
        new_image_depth = (image_depth - 127.5)/128     # [-1,1]
        
        return {'image_x': new_image_x,'image_ir': new_image_ir,'image_depth': new_image_depth, 'binary_mask': binary_mask, 'string_name': string_name}


class ToTensor_valtest(object):
    """
        Convert ndarrays in sample to Tensors.
        process only one batch every time
    """

    def __call__(self, sample):
        image_x, image_ir, image_depth, binary_mask, string_name = sample['image_x'], sample['image_ir'], sample['image_depth'],sample['binary_mask'],sample['string_name']
        
        # swap color axis because    BGR2RGB
        # numpy image: (batch_size) x T x H x W x C
        # torch image: (batch_size) x T x C X H X W
        image_x = image_x[:,:,:,::-1].transpose((0, 3, 1, 2))
        image_x = np.array(image_x)
        
        image_ir = image_ir[:,:,:,::-1].transpose((0, 3, 1, 2))
        image_ir = np.array(image_ir)
        
        image_depth = image_depth[:,:,:,::-1].transpose((0, 3, 1, 2))
        image_depth = np.array(image_depth)
                        
        binary_mask = np.array(binary_mask)
        
        return {'image_x': torch.from_numpy(image_x.astype(np.float)).float(), 'image_ir': torch.from_numpy(image_ir.astype(np.float)).float(), 'image_depth': torch.from_numpy(image_depth.astype(np.float)).float(), 'binary_mask': torch.from_numpy(binary_mask.astype(np.float)).float(), 'string_name': string_name} 



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


class Surfing_valtest(Dataset):

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
        image_path2 = os.path.join(image_path, 'rgb')
        ir_path = os.path.join(image_path, 'ir')
        depth_path = os.path.join(image_path, 'depth')
             
        image_x, image_ir, image_depth, binary_mask = self.get_single_image_x(image_path2, ir_path, depth_path, videoname)

        videoname = videoname.replace("/home/leyan/DataSet/Surfing-2020-Anti-spoofing/data/", "")
        sample = {'image_x': image_x,'image_ir': image_ir,'image_depth': image_depth, 'binary_mask': binary_mask, 'string_name': videoname}

        if self.transform:
            sample = self.transform(sample)
        sample['live'] = live
        return sample

    def get_single_image_x(self, image_path, ir_path, depth_path, videoname):

        files_total = len([name for name in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, name))])
        print("files_total: ", files_total)
        interval = files_total//frames_total
        
        image_x = np.zeros((frames_total, 256, 256, 3))
        image_ir = np.zeros((frames_total, 256, 256, 3))
        image_depth = np.zeros((frames_total, 256, 256, 3))
        
        binary_mask = np.zeros((frames_total, 32, 32))
        
        
        
        # random choose 1 frame
        image_ids = glob(f"{image_path}/*")
        image_ids.sort()
        for ii in range(frames_total):
            s = image_ids[ii]
            # image_id = ii*interval + 1 
            
            # s = "%06d.jpg" % image_id 
            # if not os.path.exists(os.path.join(image_path, s)):
            #     s = "%08d.jpg" % image_id
            s = s.replace(f"{image_path}/", "")
                 
            # 000001 
            # 00000001     
            
            # RGB
            image_path2 = os.path.join(image_path, s)
            image_x_temp = cv2.imread(image_path2)

            # depth
            image_path2_depth = os.path.join(depth_path, s).replace("jpg","png")
            image_x_temp_depth = cv2.imread(image_path2_depth)
            
            # ir
            image_path2_ir = os.path.join(ir_path, s).replace("jpg","png")

            if os.path.exists(image_path2_ir): 
                    pass
            else:
                image_path2_ir = image_path2_ir.replace("png","jpg")
                image_x_temp_depth = cv2.rotate(image_x_temp_depth, cv2.ROTATE_90_COUNTERCLOCKWISE)

            image_x_temp_ir = cv2.imread(image_path2_ir)
            
            
            
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

