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
from utils import roc_curve, get_err_threhold
import time

def mmerge_pp(rgb, ir, depth, map_x):
    rgb = rgb.cpu().data.numpy()[0]
    rgb = rgb * 128 + 127.5
    rgb = rgb.astype(np.uint8)

    ir = ir.cpu().data.numpy()[0]
    ir = ir * 128 + 127.5
    ir = ir.astype(np.uint8)

    depth = depth.cpu().data.numpy()[0]
    depth = depth * 128 + 127.5
    depth = depth.astype(np.uint8)

    rgb = rgb.transpose((1, 2, 0))
    ir = ir.transpose((1, 2, 0))
    depth = depth.transpose((1, 2, 0))

    map_x = map_x.cpu()
    heatmap = torch.zeros(map_x.size(1), map_x.size(2))
    for i in range(map_x.size(0)):
        heatmap += torch.pow(map_x[i,:,:],2).view(map_x.size(1),map_x.size(2))

    heatmap = heatmap.data.numpy()
    heatmap = resize(heatmap, depth.shape)
    # heatmap = np.stack((heatmap,)*3, axis=-1)
    heatmap = heatmap * 128 + 0

    merge_frame = np.concatenate([rgb, ir, depth, heatmap], axis=1)
    merge_frame = merge_frame.astype(np.uint8)
    return merge_frame

def perf_measure(val_labels, val_scores, thres):
    fpr,tpr,threshold = roc_curve(val_labels, val_scores, pos_label=1)
    val_err, val_threshold = get_err_threhold(fpr, tpr, threshold)    

    TP = 0
    FP = 0
    TN = 0
    FN = 0

    y_pred = [int(y > thres) for y in val_scores]

    for i in range(len(y_pred)): 
        if val_labels[i]==y_pred[i]==1:
           TP += 1
        if y_pred[i]==1 and val_labels[i]!=y_pred[i]:
           FP += 1
        if val_labels[i]==y_pred[i]==0:
           TN += 1
        if y_pred[i]==0 and val_labels[i]!=y_pred[i]:
           FN += 1

    val_APCER = FP / (TN + FP)
    val_BPCER = FN / (FN + TP)
    val_ACER = val_APCER + val_BPCER / 2
    val_ACC= (TP+TN)/(TP + TN + FP + FN)    

    return val_threshold, val_err, val_ACC, val_APCER, val_BPCER, val_ACER

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
    # model.load_state_dict(torch.load(f'./{args.log}/{args.log}_last.pkl'))
    model.load_state_dict(torch.load(f'./Multi-modal/{args.log}/{args.log}_last.pkl'))


    model = model.cuda()

    print(model) 
    


    model.eval()
    
    with torch.no_grad():
        ###########################################
        '''                val             '''
        ###########################################
        # val for threshold
        val_data = Spoofing_valtest(test_list, image_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
        dataloader_val = DataLoader(val_data, batch_size=1, shuffle=True, num_workers=4)
        
        map_score_list = []

        epoch = 50

        live = 0
        spoof=0
        
        test_scores = []
        test_labels = []

        video_save_path = "test/videos/inference.mp4"
        os.makedirs(os.path.dirname(video_save_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        size = (1280, 720)
        video_fps = 25.0
        out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        video_save_path = "test/videos/Live.mp4"
        os.makedirs(os.path.dirname(video_save_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        size = (1280, 720)
        video_fps = 25.0
        out2 = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        video_save_path = "test/videos/Spoof.mp4"
        os.makedirs(os.path.dirname(video_save_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        size = (1280, 720)
        video_fps = 25.0
        out3 = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        for i, sample_batched in enumerate(dataloader_val):
            
            logging.info(i)
            fps = 0.0
            t1 = time.time()
            # get the inputs
            inputs = sample_batched['image_x'].cuda()
            spoof_label = int(sample_batched['live'].numpy()[0])
            inputs_ir, inputs_depth = sample_batched['image_ir'].cuda(), sample_batched['image_depth'].cuda()
            string_name, binary_mask = sample_batched['string_name'], sample_batched['binary_mask'].cuda()
            sn = string_name[0].replace("/","_")
            
            
            map_score = 0.0    
            FeatureMapTList = []   
            # print(inputs.shape[1])
            for frame_t in range(inputs.shape[1]):
                orign_img = inputs[:,frame_t,:,:,:]
                map_x, embedding, (x_Block1, x_Block2, x_Block3, x_input), \
                                    (x_Block1_M2, x_Block2_M2, x_Block3_M2, x2), \
                                    (x_Block1_M3, x_Block2_M3, x_Block3_M3, x3) =  model(inputs[:,frame_t,:,:,:], inputs_ir[:,frame_t,:,:,:], inputs_depth[:,frame_t,:,:,:])                        

                score_norm = torch.sum(map_x)/torch.sum(binary_mask[:,frame_t,:,:])
                map_score += score_norm
 
                # mMap_rgb = ValidFeatureMap(orign_img, sn, x_input, x_Block1, x_Block2, x_Block3, map_x, writer, frame_t, epoch, 0, TAG="RGB")
                # mMap_ir = ValidFeatureMap(inputs_ir[:,frame_t,:,:,:], sn, x2, x_Block1_M2, x_Block2_M2, x_Block3_M2, map_x, writer, frame_t, epoch, 0, TAG="IR")
                # mMap_depth = ValidFeatureMap(inputs_depth[:,frame_t,:,:,:], sn, x3, x_Block1_M3, x_Block2_M3, x_Block3_M3, map_x, writer, frame_t, epoch, 0, TAG="DEPTH")
                # merge_heatmap = np.concatenate([mMap_rgb, mMap_ir, mMap_depth], axis=0)    
                # FeatureMapTList.append(merge_heatmap)  

                merge_frame = mmerge_pp(orign_img, inputs_ir[:,frame_t,:,:,:], inputs_depth[:,frame_t,:,:,:], map_x)
                FeatureMapTList.append(merge_frame)

                

            latency = (time.time()-t1)
            print("latency= %.2f s"%(latency))
            fps  = (1/latency)
            print("fps= %.2f"%(fps))

            map_score = map_score/inputs.shape[1]
            
            
            if map_score>1:
                map_score = 1.0
            else:
                map_score = map_score.cpu().data

            map_score_list.append('{} {}\n'.format( string_name[0], map_score ))
            test_scores.append(map_score)
            test_labels.append(spoof_label)
            titile_label = f"{string_name[0]} seq_num:{inputs.shape[1]} spoof:{spoof_label} "
            text_label = f"prob:{(map_score*100):2.3f}% latency={latency:2.2f}s fps={fps:2.2f}"

            # if (map_score > args.thres and live < 20) or (map_score < args.thres and spoof < 20):
            #     for t, heatmap in enumerate(FeatureMapTList):
            #         if map_score > args.thres:    
            #             writer.add_image(f"epoch_{epoch}_Living/map_score_{map_score}/{sn}/rgb_ir_depth", heatmap, t, dataformats="HWC")
            #         else:                       
            #             writer.add_image(f"epoch_{epoch}_Spoofing/map_score_{map_score}/{sn}/rgb_ir_depth", heatmap, t, dataformats="HWC")
            #     if map_score > args.thres: live = live + 1
            #     else: spoof = spoof + 1

            # if (live < 20) or (spoof < 20):
            #     for t, heatmap in enumerate(FeatureMapTList):
            #         if spoof_label == 1:    
            #             writer.add_image(f"epoch_{epoch}_Living/map_score_{map_score}/{sn}/rgb_ir_depth", heatmap, t, dataformats="HWC")
            #         else:                       
            #             writer.add_image(f"epoch_{epoch}_Spoofing/map_score_{map_score}/{sn}/rgb_ir_depth", heatmap, t, dataformats="HWC")
            #     if spoof_label == 1: 
            #         live = live + 1
            #     else: 
            #         spoof = spoof + 1

            # if spoof_label == 1:
            
            
            for t, heatmap in enumerate(FeatureMapTList):
                heatmap_RGB = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                heatmap_RGB = cv2.resize(heatmap_RGB, (1280, 720))
                heatmap_RGB = cv2.putText(heatmap_RGB, titile_label, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                heatmap_RGB = cv2.putText(heatmap_RGB, text_label, (0, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                out.write(heatmap_RGB)
                if spoof_label == 1: out2.write(heatmap_RGB)
                elif spoof_label == 0: out3.write(heatmap_RGB)
                cv2.imshow("w", heatmap_RGB)
                c= cv2.waitKey(1) & 0xff                 
                if c==27:
                    break
                # cv2.destroyAllWindows() 

            if spoof_label == 1: 
                live = live + 1
            else: 
                spoof = spoof + 1           


            # check specal predict err
            # if (live < 20) or (spoof < 20):
            #     for t, heatmap in enumerate(FeatureMapTList):
            #         if spoof_label == 1 and map_score < 0.9:    
            #             writer.add_image(f"epoch_{epoch}_Living/map_score_{map_score}/{sn}/rgb_ir_depth", heatmap, t, dataformats="HWC")
            #         elif spoof_label == 0 and map_score > 0.9:                       
            #             writer.add_image(f"epoch_{epoch}_Spoofing/map_score_{map_score}/{sn}/rgb_ir_depth", heatmap, t, dataformats="HWC")
            #     if spoof_label == 1 and map_score < 0.9: 
            #         live = live + 1
            #     elif spoof_label == 0 and map_score > 0.9:   
            #         spoof = spoof + 1
             
            print(f"display {live} living map, {spoof} Spoof map")

        out.release()
        out2.release()
        out3.release()
        best_threshold, _, val_ACC, val_APCER, val_BPCER, val_ACER = perf_measure(test_labels, test_scores, args.thres)

        logging.info("Epoch %d The APCER is: %f" % ((epoch + 1),val_APCER))
        logging.info("Epoch %d The BPCER is: %f" % ((epoch + 1),val_BPCER))
        logging.info("Epoch %d The ACER is: %f" % ((epoch + 1),val_ACER))
        logging.info("Epoch %d The Accuracy is: %f" % ((epoch + 1),val_ACC))  
        logging.info("Epoch %d The Best threshold is: %f" % ((epoch + 1),best_threshold))  

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
    parser.add_argument('--log', type=str, default="CDCN_3modality_BALL_GRAD2", help='log and save model name')
    parser.add_argument('--protocol', type=str, default="4@1", help='log and save model name')
    # parser.add_argument('--finetune', action='store_true', default=False, help='whether finetune other models')
    parser.add_argument('--theta', type=float, default=0.7, help='hyper-parameters in CDCNpp')
    parser.add_argument('--thres', type=float, default=0.97, help='hyper-parameters in CDCNpp')
    
    args = parser.parse_args()

    init_logging(0, args.log, name="inference.log")

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
    # test_list = f'{image_dir}/{args.protocol}_test_ref.txt'
    # test_list = f'{image_dir}/{args.protocol}_test_res.txt'
    # test_list = f'{test_dir}/4@2_test_res.txt'
    # test_list = f'{test_dir}/4@3_test_res.txt'
    test_list = f"{image_dir}/4@3_test_ref.txt"
    train_test()
