from __future__ import print_function, division
import torch
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import argparse,os
import pandas as pd
import cv2
import numpy as np
import random
import math
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms



from models.CDCNs import Conv2d_cd, CDCN_3modality2

from Loadtemporal_valtest_3modality import Spoofing_valtest, Normaliztion_valtest, ToTensor_valtest


import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import copy
import pdb

import logging
from utils import init_logging

from utils import AvgrageMeter, accuracy, performances, performances_SiWM_EER, test_threshold_based
from tensorboardX import SummaryWriter
from skimage.transform import resize, rescale



# feature  -->   [ batch, channel, height, width ]
def ValidFeatureMap( orign_img, sn, x, feature1, feature2, feature3, map_x, writer, step, epoch, spoof_label, val=False,TAG="RGB"):
    orign_img = orign_img.cpu().data.numpy()[0]
    orign_img = orign_img * 128 + 127.5
    orign_img = orign_img.astype(np.uint8)
    if val: 
        writer.add_image(f"epoch_{epoch}/orign_img_{sn}", orign_img, step, dataformats="CHW")
    ## initial images 
    feature_first_frame = x[0,:,:,:].cpu()    ## the middle frame 

    heatmap = torch.zeros(feature_first_frame.size(1), feature_first_frame.size(2))
    for i in range(feature_first_frame.size(0)):
        heatmap += torch.pow(feature_first_frame[i,:,:],2).view(feature_first_frame.size(1),feature_first_frame.size(2))

    heatmap1 = heatmap.data.numpy()
    if val: writer.add_image(f"epoch_{epoch}/x_visual_{sn}", heatmap1, step, dataformats="HW")
    fig = plt.figure() 
    ax = fig.add_subplot(111)
    plt.imshow(heatmap1)
    plt.colorbar()
    plt.savefig(args.log+'/'+ 'x_visual.jpg')
    plt.close()


    ## first feature
    feature_first_frame = feature1[0,:,:,:].cpu()    ## the middle frame 
    heatmap = torch.zeros(feature_first_frame.size(1), feature_first_frame.size(2))
    for i in range(feature_first_frame.size(0)):
        heatmap += torch.pow(feature_first_frame[i,:,:],2).view(feature_first_frame.size(1),feature_first_frame.size(2))

    heatmap2 = heatmap.data.numpy()
    if val: writer.add_image(f"epoch_{epoch}/x_Block1_visual_{sn}", heatmap2, step, dataformats="HW")
    
    fig = plt.figure() 
    ax = fig.add_subplot(111)
    plt.imshow(heatmap2)
    plt.colorbar()
    plt.savefig(args.log+'/'+ 'x_Block1_visual.jpg')
    plt.close()
    
    ## second feature
    feature_first_frame = feature2[0,:,:,:].cpu()    ## the middle frame 
    heatmap = torch.zeros(feature_first_frame.size(1), feature_first_frame.size(2))
    for i in range(feature_first_frame.size(0)):
        heatmap += torch.pow(feature_first_frame[i,:,:],2).view(feature_first_frame.size(1),feature_first_frame.size(2))

    heatmap3 = heatmap.data.numpy()
    if val: writer.add_image(f"epoch_{epoch}/x_Block2_visual_{sn}", heatmap3, step, dataformats="HW")

    
    fig = plt.figure() 
    ax = fig.add_subplot(111)
    plt.imshow(heatmap3)
    plt.colorbar()
    plt.savefig(args.log+'/'+ 'x_Block2_visual.jpg')
    plt.close()
    
    ## third feature
    feature_first_frame = feature3[0,:,:,:].cpu()    ## the middle frame 
    heatmap = torch.zeros(feature_first_frame.size(1), feature_first_frame.size(2))
    for i in range(feature_first_frame.size(0)):
        heatmap += torch.pow(feature_first_frame[i,:,:],2).view(feature_first_frame.size(1),feature_first_frame.size(2))

    heatmap4 = heatmap.data.numpy()
    if val: writer.add_image(f"epoch_{epoch}/x_Block3_visual_{sn}", heatmap4, step, dataformats="HW")
    
    fig = plt.figure() 
    ax = fig.add_subplot(111)
    plt.imshow(heatmap4)
    plt.colorbar()
    plt.savefig(args.log+'/'+ 'x_Block3_visual.jpg')
    plt.close()
    
    ## third feature
    heatmap5 = torch.pow(map_x[0,:,:],2)    ## the middle frame 

    heatmap5 = heatmap5.data.cpu().numpy()
    if val: writer.add_image(f"epoch_{epoch}/x_DepthMap_visual_{sn}", heatmap5, step, dataformats="HW")
    
    fig = plt.figure() 
    ax = fig.add_subplot(111)
    plt.imshow(heatmap5)
    plt.colorbar()
    plt.savefig(args.log+'/'+ 'x_DepthMap_visual.jpg')
    plt.close()

    # merge
    heatmap2 = resize(heatmap2, heatmap1.shape)
    heatmap3 = resize(heatmap3, heatmap1.shape)
    heatmap4 = resize(heatmap4, heatmap1.shape)
    heatmap5 = resize(heatmap5, heatmap1.shape)

    heatmap1 = np.stack((heatmap1,)*3, axis=-1)
    heatmap2 = np.stack((heatmap2,)*3, axis=-1)
    heatmap3 = np.stack((heatmap3,)*3, axis=-1)
    heatmap4 = np.stack((heatmap4,)*3, axis=-1)
    heatmap5 = np.stack((heatmap5,)*3, axis=-1)

    heatmap1 = heatmap1 * 128 + 0
    heatmap2 = heatmap2 * 128 + 0
    heatmap3 = heatmap3 * 128 + 0
    heatmap4 = heatmap4 * 128 + 0
    heatmap5 = heatmap5 * 128 + 0

    orign_img = orign_img.transpose((1, 2, 0))

    merge_heatmap = np.concatenate([orign_img, heatmap1, heatmap2, heatmap3, heatmap4, heatmap5], axis=1)
    merge_heatmap = merge_heatmap.astype(np.uint8)

    # if spoof_label == 0:
    #     writer.add_image(f"epoch_{epoch}/Valid_Spoofing_{sn}/{TAG}", merge_heatmap, step, dataformats="HWC")
    # else:
    #     writer.add_image(f"epoch_{epoch}/Valid_Living_{sn}/{TAG}", merge_heatmap, step, dataformats="HWC")
    
    return merge_heatmap

# main function
def train_test():


    print("test:\n ")
    writer = SummaryWriter(log_dir=os.path.join(args.log, "Predict"))

     
    model = CDCN_3modality2( basic_conv=Conv2d_cd, theta=args.theta)
    
    # model.load_state_dict(torch.load('CDCN_3modality2_P1/CDCN_3modality2_P1_50.pkl'))
    model.load_state_dict(torch.load(f'./{args.log}/{args.log}_last.pkl'))


    model = model.cuda()

    print(model) 
    


    model.eval()
    
    with torch.no_grad():
        ###########################################
        '''                val             '''
        ###########################################
        # val for threshold
        val_data = Spoofing_valtest(test_list, image_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
        dataloader_val = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=4)
        
        map_score_list = []

        epoch = 50

        live = 0
        spoof=0
        
        for i, sample_batched in enumerate(dataloader_val):
            
            logging.info(i)
            
            # get the inputs
            inputs = sample_batched['image_x'].cuda()
            inputs_ir, inputs_depth = sample_batched['image_ir'].cuda(), sample_batched['image_depth'].cuda()
            string_name, binary_mask = sample_batched['string_name'], sample_batched['binary_mask'].cuda()
            sn = string_name[0].replace("/","_")
            
            
            map_score = 0.0
            FeatureMapTList_RGB = []   
            FeatureMapTList_IR = []   
            FeatureMapTList_Depth = []
            FeatureMapTList = []   

            for frame_t in range(inputs.shape[1]):
                orign_img = inputs[:,frame_t,:,:,:]
                map_x, embedding, (x_Block1, x_Block2, x_Block3, x_input), \
                                    (x_Block1_M2, x_Block2_M2, x_Block3_M2, x2), \
                                    (x_Block1_M3, x_Block2_M3, x_Block3_M3, x3) =  model(inputs[:,frame_t,:,:,:], inputs_ir[:,frame_t,:,:,:], inputs_depth[:,frame_t,:,:,:])                        

                score_norm = torch.sum(map_x)/torch.sum(binary_mask[:,frame_t,:,:])
                map_score += score_norm
 
                # if i < 10 :
                if live < 10 or spoof < 10:
                    mMap_rgb = ValidFeatureMap(orign_img, sn, x_input, x_Block1, x_Block2, x_Block3, map_x, writer, frame_t, epoch, 0, TAG="RGB")
                    mMap_ir = ValidFeatureMap(inputs_ir[:,frame_t,:,:,:], sn, x2, x_Block1_M2, x_Block2_M2, x_Block3_M2, map_x, writer, frame_t, epoch, 0, TAG="IR")
                    mMap_depth = ValidFeatureMap(inputs_depth[:,frame_t,:,:,:], sn, x3, x_Block1_M3, x_Block2_M3, x_Block3_M3, map_x, writer, frame_t, epoch, 0, TAG="DEPTH")
                    merge_heatmap = np.concatenate([mMap_rgb, mMap_ir, mMap_depth], axis=0)    
                    # writer.add_image(f"epoch_{epoch}/{sn}/rgb_ir_depth", merge_heatmap, frame_t, dataformats="HWC")  
                    FeatureMapTList.append(merge_heatmap)           

            map_score = map_score/inputs.shape[1]
            
            
            if map_score>1:
                map_score = 1.0
            else:
                map_score = map_score.cpu().data

            map_score_list.append('{} {}\n'.format( string_name[0], map_score ))

            if (map_score > args.thres and live < 10) or (map_score < args.thres and spoof < 10):
                for t, heatmap in enumerate(FeatureMapTList):
                    if map_score > args.thres:    
                        writer.add_image(f"epoch_{epoch}_Living/map_score_{map_score}/{sn}/rgb_ir_depth", heatmap, t, dataformats="HWC")
                    else:                       
                        writer.add_image(f"epoch_{epoch}_Spoofing/map_score_{map_score}/{sn}/rgb_ir_depth", heatmap, t, dataformats="HWC")
                if map_score > args.thres: live = live + 1
                else: spoof = spoof + 1


                # for t, heatmap in enumerate(FeatureMapTList):
                #     if map_score > args.thres:
                #         if live > 10: continue
                #         live = live + 1
                #         writer.add_image(f"epoch_{epoch}_Living/map_score_{map_score}/{sn}/rgb_ir_depth", heatmap, t, dataformats="HWC")
                #     else:
                #         if spoof > 10: continue
                #         spoof = spoof + 1
                #         writer.add_image(f"epoch_{epoch}_Spoofing/map_score_{map_score}/{sn}/rgb_ir_depth", heatmap, t, dataformats="HWC")

            print(f"display {live} living map, {spoof} Spoof map")
        map_score_val_filename = "./" + args.log+'/'+ args.log+ '_map_score_test_50.txt'
        with open(map_score_val_filename, 'w') as file:
            file.writelines(map_score_list) 
                
    logging.info('Finished testing')

    
  

  
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="save quality using landmarkpose model")
    parser.add_argument('--gpu', type=int, default=3, help='the gpu id used for predict')
    # parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')  #default=0.0001
    # parser.add_argument('--batchsize', type=int, default=9, help='initial batchsize')  #default=7  
    # parser.add_argument('--step_size', type=int, default=20, help='how many epochs lr decays once')  # 500  | DPC = 400
    # parser.add_argument('--gamma', type=float, default=0.5, help='gamma of optim.lr_scheduler.StepLR, decay of lr')
    # parser.add_argument('--echo_batches', type=int, default=50, help='how many batches display once')  # 50
    # parser.add_argument('--epochs', type=int, default=50, help='total training epochs')
    parser.add_argument('--log', type=str, default="CDCN_3modality_P1_4@1", help='log and save model name')
    parser.add_argument('--protocol', type=str, default="4@1", help='log and save model name')
    # parser.add_argument('--finetune', action='store_true', default=False, help='whether finetune other models')
    parser.add_argument('--theta', type=float, default=0.7, help='hyper-parameters in CDCNpp')
    parser.add_argument('--thres', type=float, default=0.97, help='hyper-parameters in CDCNpp')
    
    args = parser.parse_args()

    init_logging(0, args.log, name="testing.log")

    logging.info('===Options==') 
    d=vars(args)

    for key, value in d.items():
        num_space = 25 - len(key)
        try:
            logging.info(": " + key + " " * num_space + str(value))
        except Exception as e:
            print(e)

    # Dataset root      
    image_dir = '/home/leyan/DataSet/CASIA-CeFA/phase2'  
    test_list = f'{image_dir}/{args.protocol}_test_res.txt'
    # test_list = f'{test_dir}/4@2_test_res.txt'
    # test_list = f'{test_dir}/4@3_test_res.txt'
    train_test()
