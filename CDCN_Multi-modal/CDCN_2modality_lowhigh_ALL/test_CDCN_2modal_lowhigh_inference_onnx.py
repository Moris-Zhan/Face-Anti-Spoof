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



from models.CDCNs import Conv2d_cd, CDCN_2modality_lowhigh

from Loadtemporal_valtest_3modality import Spoofing_valtest, Normaliztion_valtest, ToTensor_valtest


import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import copy
import pdb

from utils import init_logging

from utils import AvgrageMeter, accuracy, performances, performances_SiWM_EER, test_threshold_based
from tensorboardX import SummaryWriter
from skimage.transform import resize, rescale
from utils import roc_curve, get_err_threhold
import time
import onnxruntime
import onnx

# Define input and outputs names, which are required to properly define
# dynamic axes
input_names = ['input.1','input.2','input.3']
output_names = ['map_x']

"""
test check onnx inference time & export onnx
"""

# main function
def train_test():


    print("test:\n ")
    writer = SummaryWriter(log_dir=os.path.join(args.log, "Predict"))

     
    model = CDCN_2modality_lowhigh( basic_conv=Conv2d_cd, theta=args.theta)
    
    # model.load_state_dict(torch.load('CDCN_2modality_lowhigh_P1/CDCN_2modality_lowhigh_P1_50.pkl'))
    model.load_state_dict(torch.load(f'./{args.log}/{args.log}_last.pkl'))
    # model.load_state_dict(torch.load(f'Multi-modal/{args.log}/{args.log}_last.pkl'))
    # model.load_state_dict(torch.load(f'./Multi-modal/{args.log}/{args.log}_last.pkl'))


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

        epoch = 100

        live = 0
        spoof=0
        
        test_scores = []
        test_labels = []

        # video_save_path = f"{args.log}/videos/inference.mp4"
        # print(video_save_path)
        # os.makedirs(os.path.dirname(video_save_path), exist_ok=True)
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # size = (1280, 720)
        # video_fps = 25.0
        # out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        # video_save_path = f"{args.log}/videos/Live.mp4"
        # os.makedirs(os.path.dirname(video_save_path), exist_ok=True)
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # size = (1280, 720)
        # video_fps = 25.0
        # out2 = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        # video_save_path = f"{args.log}/videos/Spoof.mp4"
        # os.makedirs(os.path.dirname(video_save_path), exist_ok=True)
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # size = (1280, 720)
        # video_fps = 25.0
        # out3 = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        for i, sample_batched in enumerate(dataloader_val):
            
            # get the inputs
            inputs = sample_batched['image_x'].cuda() # [256, 256]
            spoof_label = int(sample_batched['live'].numpy()[0])
            inputs_ir, inputs_depth = sample_batched['image_ir'].cuda(), sample_batched['image_depth'].cuda()
            string_name, binary_mask = sample_batched['string_name'], sample_batched['binary_mask'].cuda()

            frame_t = 0
            
            


            """Transform ONNX"""           
            ori_output_file = f"{args.log}/mcdcn_ori.onnx"
            output_file = f"{args.log}/mcdcn_2modal_lohi.onnx"
            if not os.path.exists(output_file):
                orign_img = inputs[:,frame_t,:,:,:]
                map_x, embedding, (x_Block1, x_Block2, x_Block3, x_input), \
                                    (x_Block1_M3, x_Block2_M3, x_Block3_M3, x3) =  model(inputs[:,frame_t,:,:,:], inputs_ir[:,frame_t,:,:,:], inputs_depth[:,frame_t,:,:,:])                        
                score_norm = torch.sum(map_x)/torch.sum(binary_mask[:,frame_t,:,:])
                torch.onnx.export(
                    model,
                    {'x1':inputs[:,frame_t,:,:,:],'x2':inputs_ir[:,frame_t,:,:,:],'x3':inputs_depth[:,frame_t,:,:,:]},
                    ori_output_file,
                    keep_initializers_as_inputs=False,
                    verbose=False,
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=None,
                    opset_version=11)

                simplify = True
                if simplify:
                    model = onnx.load(ori_output_file)
                    if simplify:
                        # !pip install onnx-simplifier==0.3.10
                        from onnxsim import simplify
                        #print(model.graph.input[0])
                        model, check = simplify(model)
                        assert check, "Simplified ONNX model could not be validated"
                    onnx.save(model, output_file)
                    os.remove(ori_output_file)

            """ONNX Inference"""
            inputs = inputs.cpu().numpy()
            inputs_ir = inputs_ir.cpu().numpy()
            inputs_depth = inputs_depth.cpu().numpy()

            session = onnxruntime.InferenceSession(output_file, None)
            input1_name = session.get_inputs()[0].name
            # input2_name = session.get_inputs()[1].name
            input3_name = session.get_inputs()[1].name
            st = time.time()
            net_outs = session.run(None, 
                    {
                        input1_name: inputs[:,frame_t,:,:,:],
                        # input2_name: inputs_ir[:,frame_t,:,:,:],
                        input3_name: inputs_depth[:,frame_t,:,:,:]
                    },)
            map_x = net_outs[0]
            binary_mask = binary_mask.cpu().numpy()
            score_norm = np.sum(map_x)/np.sum(binary_mask[:,frame_t,:,:])
            print("score_norm: ", score_norm)
            latency = time.time()-st
            print(f"latency: {latency} s")

            rgb = inputs[:,frame_t,:,:,:].squeeze().transpose(1, 2, 0)
            ir = inputs_ir[:,frame_t,:,:,:].squeeze().transpose(1, 2, 0)
            depth = inputs_depth[:,frame_t,:,:,:].squeeze().transpose(1, 2, 0)

            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            ir = cv2.cvtColor(ir, cv2.COLOR_BGR2RGB)
            depth = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)

            img = np.hstack([
                rgb, 
                ir, 
                depth
                , ])
            # cv2.imshow("", img)
            # cv2.waitKey(0)
            # break


  
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="save quality using landmarkpose model")
    parser.add_argument('--gpu', type=int, default=3, help='the gpu id used for predict')
    # parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')  #default=0.0001
    # parser.add_argument('--batchsize', type=int, default=9, help='initial batchsize')  #default=7  
    # parser.add_argument('--step_size', type=int, default=20, help='how many epochs lr decays once')  # 500  | DPC = 400
    # parser.add_argument('--gamma', type=float, default=0.5, help='gamma of optim.lr_scheduler.StepLR, decay of lr')
    # parser.add_argument('--echo_batches', type=int, default=50, help='how many batches display once')  # 50
    # parser.add_argument('--epochs', type=int, default=50, help='total training epochs')
    parser.add_argument('--log', type=str, default="CDCN_2modality_lowhigh_ALL", help='log and save model name')
    # parser.add_argument('--finetune', action='store_true', default=False, help='whether finetune other models')
    parser.add_argument('--theta', type=float, default=0.7, help='hyper-parameters in CDCNpp')
    parser.add_argument('--thres', type=float, default=0.97, help='hyper-parameters in CDCNpp')
    
    args = parser.parse_args()

    d=vars(args)    

    # Dataset root      
    image_dir = '/home/leyan/DataSet/CASIA-CeFA/phase2'    
    test_list = f"{image_dir}/4@3_test_ref.txt"
    train_test()
