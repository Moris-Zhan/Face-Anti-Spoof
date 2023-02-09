from __future__ import print_function, division
import torch
import matplotlib as mpl
mpl.use('TkAgg')
import argparse,os
import pandas as pd
import cv2
import numpy as np
import random
import math
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


from models.CDCNs import Conv2d_cd, CDCN_3modality2


from Loadtemporal_BinaryMask_train_3modality import Spoofing_train, Normaliztion, ToTensor, RandomHorizontalFlip, Cutout, RandomErasing
from Loadtemporal_valtest_3modality import Spoofing_valtest, Normaliztion_valtest, ToTensor_valtest
from tensorboardX import SummaryWriter

import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import copy
import pdb
import io

from utils import AvgrageMeter, accuracy, performances
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
from torchinfo import summary
import logging
from utils import init_logging, get_threshold
from skimage.transform import resize
# Dataset root
image_dir = '/home/leyan/DataSet/CASIA-CeFA/phase1'         
   
train_list = f"{image_dir}/4@1_train.txt"
val_list = f'{image_dir}/4@1_dev_ref.txt'

#train_list = f'{image_dir}/4@2_train.txt'
#val_list = f'{image_dir}/4@2_dev_ref.txt'

#train_list = f'{image_dir}/4@3_train.txt'
#val_list = f'{image_dir}/4@3_dev_ref.txt'

def perf_measure(y_actual, y_pred, thres):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    y_actual = [int(y) for y in y_actual]
    y_pred = [int(y > thres) for y in y_pred]

    for i in range(len(y_pred)): 
        if y_actual[i]==y_pred[i]==1:
           TP += 1
        if y_pred[i]==1 and y_actual[i]!=y_pred[i]:
           FP += 1
        if y_actual[i]==y_pred[i]==0:
           TN += 1
        if y_pred[i]==0 and y_actual[i]!=y_pred[i]:
           FN += 1

    APCER = FP / (TN + FP)
    BPCER = FN / (FN + TP)
    ACER = APCER + BPCER / 2
    accuracy= (TP+TN)/(TP + TN + FP + FN)

    return(TP, FP, TN, FN), (APCER, BPCER, ACER), accuracy
 
 # feature  -->   [ batch, channel, height, width ]
def TFeatureMap2Heatmap( orign_img, sn, x, feature1, feature2, feature3, map_x, writer, epoch, val=True):
    orign_img = orign_img.cpu().data.numpy()[0]
    orign_img = orign_img * 128 + 127.5
    orign_img = orign_img.astype(np.uint8)
    if val: 
        writer.add_image(f"epoch_{epoch}/orign_img", orign_img, 0, dataformats="CHW")
    ## initial images 
    feature_first_frame = x[0,:,:,:].cpu()    ## the middle frame 

    heatmap = torch.zeros(feature_first_frame.size(1), feature_first_frame.size(2))
    for i in range(feature_first_frame.size(0)): # sum three channel weight featur map
        heatmap += torch.pow(feature_first_frame[i,:,:],2).view(feature_first_frame.size(1),feature_first_frame.size(2))

    heatmap1 = heatmap.data.numpy()
    if val: writer.add_image(f"epoch_{epoch}/x_visual", heatmap1, 0, dataformats="HW")
    

    ## first feature
    feature_first_frame = feature1[0,:,:,:].cpu()    ## the middle frame 
    heatmap = torch.zeros(feature_first_frame.size(1), feature_first_frame.size(2))
    for i in range(feature_first_frame.size(0)):
        heatmap += torch.pow(feature_first_frame[i,:,:],2).view(feature_first_frame.size(1),feature_first_frame.size(2))

    heatmap2 = heatmap.data.numpy()
    if val: writer.add_image(f"epoch_{epoch}/x_Block1_visual", heatmap2, 0, dataformats="HW")   
    
    ## second feature
    feature_first_frame = feature2[0,:,:,:].cpu()    ## the middle frame 
    heatmap = torch.zeros(feature_first_frame.size(1), feature_first_frame.size(2))
    for i in range(feature_first_frame.size(0)):
        heatmap += torch.pow(feature_first_frame[i,:,:],2).view(feature_first_frame.size(1),feature_first_frame.size(2))

    heatmap3 = heatmap.data.numpy()
    if val: writer.add_image(f"epoch_{epoch}/x_Block2_visual", heatmap3, 0, dataformats="HW")   
    
    ## third feature
    feature_first_frame = feature3[0,:,:,:].cpu()    ## the middle frame 
    heatmap = torch.zeros(feature_first_frame.size(1), feature_first_frame.size(2))
    for i in range(feature_first_frame.size(0)):
        heatmap += torch.pow(feature_first_frame[i,:,:],2).view(feature_first_frame.size(1),feature_first_frame.size(2))

    heatmap4 = heatmap.data.numpy()
    if val: writer.add_image(f"epoch_{epoch}/x_Block3_visual", heatmap4, 0, dataformats="HW")    
    
    ## third feature
    heatmap5 = torch.pow(map_x[0,:,:],2)    ## the middle frame 

    heatmap5 = heatmap5.data.cpu().numpy()
    if val: writer.add_image(f"epoch_{epoch}/x_DepthMap_visual", heatmap5, 0, dataformats="HW")
   
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
    writer.add_image(f"epoch_{epoch}/merge_heatmap_{sn}", merge_heatmap, 0, dataformats="HWC")

# feature  -->   [ batch, channel, height, width ]
def FeatureMap2Heatmap( orign_img, sn, x, feature1, feature2, feature3, map_x, writer, step, epoch, val=False):
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
    writer.add_image(f"epoch_{epoch}/merge_heatmap_{sn}", merge_heatmap, step, dataformats="HWC")
    
    return merge_heatmap





def contrast_depth_conv(input):
    ''' compute contrast depth in both of (out, label) '''
    '''
        input  32x32
        output 8x32x32
    '''
    

    kernel_filter_list =[
                        [[1,0,0],[0,-1,0],[0,0,0]], [[0,1,0],[0,-1,0],[0,0,0]], [[0,0,1],[0,-1,0],[0,0,0]],
                        [[0,0,0],[1,-1,0],[0,0,0]], [[0,0,0],[0,-1,1],[0,0,0]],
                        [[0,0,0],[0,-1,0],[1,0,0]], [[0,0,0],[0,-1,0],[0,1,0]], [[0,0,0],[0,-1,0],[0,0,1]]
                        ]
    
    kernel_filter = np.array(kernel_filter_list, np.float32)
    
    kernel_filter = torch.from_numpy(kernel_filter.astype(np.float)).float().cuda()
    # weights (in_channel, out_channel, kernel, kernel)
    kernel_filter = kernel_filter.unsqueeze(dim=1)
    
    input = input.unsqueeze(dim=1).expand(input.shape[0], 8, input.shape[1],input.shape[2])
    
    contrast_depth = F.conv2d(input, weight=kernel_filter, groups=8)  # depthwise conv
    
    return contrast_depth


class Contrast_depth_loss(nn.Module):    # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    def __init__(self):
        super(Contrast_depth_loss,self).__init__()
        return
    def forward(self, out, label): 
        '''
        compute contrast depth in both of (out, label),
        then get the loss of them
        tf.atrous_convd match tf-versions: 1.4
        '''
        contrast_out = contrast_depth_conv(out)
        contrast_label = contrast_depth_conv(label)
        
        
        criterion_MSE = nn.MSELoss().cuda()
    
        loss = criterion_MSE(contrast_out, contrast_label)
        #loss = torch.pow(contrast_out - contrast_label, 2)
        #loss = torch.mean(loss)
    
        return loss




# main function
def train_test():
    # GPU  & log file  -->   if use DataParallel, please comment this command
    #os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % (args.gpu)
    
    echo_batches = args.echo_batches
    logging.info("Oulu-NPU, P1:\n ")

    writer = SummaryWriter(log_dir=os.path.join(args.log, "tensorboard"))

    # load the network, load the pre-trained model in UCF101?
    finetune = args.finetune
    if finetune==True:
        logging.info('finetune!\n')

    else:
        logging.info("train from scratch!\n")         
        model = CDCN_3modality2( basic_conv=Conv2d_cd, theta=args.theta)
        # -------------------------------------------------------------------------------    
        IM_SHAPE = (256, 256, 3)    
        f = io.StringIO()
        with redirect_stdout(f):        
            summary(model, [(args.batchsize, IM_SHAPE[2], IM_SHAPE[0], IM_SHAPE[1])] * 3 )
        lines = f.getvalue()

        with open( os.path.join(args.log, "model.txt") ,"w") as f:
            [f.write(line) for line in lines]
        print(lines)
        # ------------------------------------------------------------------------------ 
        rndm_input = torch.autograd.Variable(torch.rand(1, 3, 256, 256), requires_grad = False).cuda()
        writer.add_graph(model, [rndm_input]*3)

        

        model = model.cuda()


        lr = args.lr
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.00005)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)    
    
    criterion_absolute_loss = nn.MSELoss().cuda()
    criterion_contrastive_loss = Contrast_depth_loss().cuda() 
    


    ACER_save = 1.0
    best_acc = 0.0
    
    for epoch in range(args.epochs):  # loop over the dataset multiple times
        scheduler.step()
        if (epoch + 1) % args.step_size == 0:
            lr *= args.gamma

        
        loss_absolute = AvgrageMeter()
        loss_contra =  AvgrageMeter()
        #top5 = utils.AvgrageMeter()
        
        
        ###########################################
        '''                train             '''
        ###########################################
        model.train()
        
        # load random 16-frame clip data every epoch
        train_data = Spoofing_train(train_list, image_dir, transform=transforms.Compose([RandomErasing(), RandomHorizontalFlip(),  ToTensor(), Cutout(), Normaliztion()]))
        dataloader_train = DataLoader(train_data, batch_size=args.batchsize, shuffle=True, num_workers=4)
        nums_train = len(dataloader_train)

        for i, sample_batched in enumerate(dataloader_train):
            # get the inputs
            inputs, binary_mask, spoof_label = sample_batched['image_x'].cuda(), sample_batched['binary_mask'].cuda(), sample_batched['spoofing_label'].cuda() 
            inputs_ir, inputs_depth = sample_batched['image_ir'].cuda(), sample_batched['image_depth'].cuda()
            sn = sample_batched['string_name'][0].replace("/","_")
            
            optimizer.zero_grad()        

            # forward + backward + optimize
            map_x, embedding, x_Block1, x_Block2, x_Block3, x_input =  model(inputs, inputs_ir, inputs_depth)

            #pdb.set_trace()
            absolute_loss = criterion_absolute_loss(map_x, binary_mask)
            contrastive_loss = criterion_contrastive_loss(map_x, binary_mask)
            
            loss =  absolute_loss + contrastive_loss
             
            loss.backward()
            
            optimizer.step()
            
            n = inputs.size(0)
            loss_absolute.update(absolute_loss.data, n)
            loss_contra.update(contrastive_loss.data, n)

            writer.add_scalar('Step/Abs_Loss', absolute_loss.data, epoch * nums_train + i)
            writer.add_scalar('Step/Contrast_Loss', contrastive_loss.data, epoch * nums_train + i)  
                        
            if i % echo_batches == echo_batches-1:    # print every 50 mini-batches
                mMap = TFeatureMap2Heatmap(inputs, sn, x_input, x_Block1, x_Block2, x_Block3, map_x, writer, epoch)
                # log written
                logging.info('epoch:%d, mini-batch:%3d, lr=%f, Absolute_Depth_loss= %.4f, Contrastive_Depth_loss= %.4f' % (epoch + 1, i + 1, lr,  loss_absolute.avg, loss_contra.avg))
        
            # break            
            
        # whole epoch average
        logging.info("epoch:%d, Train: Absolute_Depth_loss= %.4f, Contrastive_Depth_loss= %.4f \n" % (epoch + 1, loss_absolute.avg, loss_contra.avg))
        writer.add_scalar('Train/avg_loss_absolute', loss_absolute.avg, epoch + 1)
        writer.add_scalar('Train/avg_loss_contra', loss_contra.avg, epoch + 1)  
    
            
        epoch_test = 1
        if epoch>10 and epoch % epoch_test == epoch_test-1:   
        # if epoch>-1 and epoch % epoch_test == epoch_test-1:  
            model.eval()
            
            with torch.no_grad():
                ###########################################
                '''                val             '''
                ###########################################
                # val for threshold
                val_data = Spoofing_valtest(val_list, image_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
                dataloader_val = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=4)
                
                map_score_list = []

                y_list = []
                y_bar_list = []
                
                for i, sample_batched in enumerate(dataloader_val):
                    # get the inputs
                    inputs = sample_batched['image_x'].cuda()
                    inputs_ir, inputs_depth = sample_batched['image_ir'].cuda(), sample_batched['image_depth'].cuda()
                    string_name, binary_mask = sample_batched['string_name'], sample_batched['binary_mask'].cuda()
                    sn = string_name[0].replace("/","_")
                    live = int(sample_batched['live'].numpy()[0])
                    optimizer.zero_grad()
                                        
                    map_score = 0.0
                    FeatureMapTList = []                    
                    for t, frame_t in enumerate (range(inputs.shape[1])):
                        orign_img = inputs[:,frame_t,:,:,:]
                        map_x, embedding, x_Block1, x_Block2, x_Block3, x_input =  model(inputs[:,frame_t,:,:,:], inputs_ir[:,frame_t,:,:,:], inputs_depth[:,frame_t,:,:,:])                        

                        score_norm = torch.sum(map_x)/torch.sum(binary_mask[:,frame_t,:,:])
                        map_score += score_norm

                        if i < 10:
                            mMap = FeatureMap2Heatmap(orign_img, sn, x_input, x_Block1, x_Block2, x_Block3, map_x, writer, t, epoch)
                            FeatureMapTList.append(mMap)  
                        else: continue

                    map_score = map_score/inputs.shape[1]
                    
                    if i < 10:
                        frame_t_heatmap = np.concatenate(FeatureMapTList, axis=0) 
                        writer.add_image("live%s_heatmap/%s"%(live, string_name[0]), frame_t_heatmap, epoch, dataformats="HWC")                        
                    
                    if map_score>1:
                        map_score = 1.0
                    else:
                        map_score = float(map_score.cpu().numpy())
    
                    map_score_list.append('{} {} {}\n'.format( string_name[0], map_score, live))
                    y_list.append(map_score)
                    y_bar_list.append(live)

                    
                map_score_val_filename = args.log+'/' + 'map_score_val_%d.txt'% (epoch + 1)
                with open(map_score_val_filename, 'w') as file:
                    file.writelines(map_score_list)                
                

            _, (APCER, BPCER, ACER), accuracy = perf_measure(y_bar_list, y_list, args.thres)

            writer.add_scalar('Validation/APCER', APCER, epoch + 1)
            writer.add_scalar('Validation/BPCER', BPCER, epoch + 1)
            writer.add_scalar('Validation/ACER', ACER, epoch + 1)
            writer.add_scalar('Validation/Accuracy', accuracy, epoch + 1)

            logging.info("The APCER is: %s" % (APCER))
            logging.info("The BPCER is: %s" % (BPCER))
            logging.info("The ACER is: %s" % (ACER))
            logging.info("The Accuracy is: %s" % (accuracy))

            if ACER > best_acc:
                best_acc = ACER
                # save the best model 
                torch.save(model.state_dict(), args.log+'/' + args.log+'_best.pkl')
                logging.info("Epoch %d The Accuracy is: %s"%((epoch + 1), ACER))
            # save the model until the next improvement     
            torch.save(model.state_dict(), args.log+'/' + args.log+ '_last.pkl')


    logging.info('Finished Training')
  

  
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="save quality using landmarkpose model")
    parser.add_argument('--gpu', type=int, default=3, help='the gpu id used for predict')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')  #default=0.0001
    parser.add_argument('--batchsize', type=int, default=4, help='initial batchsize')  #default=9  
    parser.add_argument('--step_size', type=int, default=20, help='how many epochs lr decays once')  # 500  | DPC = 400
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma of optim.lr_scheduler.StepLR, decay of lr')
    parser.add_argument('--echo_batches', type=int, default=1000, help='how many batches display once')  # 50
    parser.add_argument('--epochs', type=int, default=50, help='total training epochs')
    parser.add_argument('--log', type=str, default="CDCN_3modality_P1_Test", help='log and save model name')
    parser.add_argument('--finetune', action='store_true', default=False, help='whether finetune other models')
    parser.add_argument('--theta', type=float, default=0.7, help='hyper-parameters in CDCNpp')
    parser.add_argument('--thres', type=float, default=0.5, help='hyper-parameters in CDCNpp')
    
    args = parser.parse_args()

    init_logging(0, args.log)

    logging.info('===Options==') 
    d=vars(args)

    # with open('commandline_args.txt', 'w') as f:        
    #     for key, value in d.items():
    #         num_space = 25 - len(key)
    #         try:
    #             f.write(key + " = " + str(value) + "\n")
    #         except Exception as e :
    #             pass

    for key, value in d.items():
        num_space = 25 - len(key)
        try:
            logging.info(": " + key + " " * num_space + str(value))
        except Exception as e:
            print(e)
            
    train_test()
