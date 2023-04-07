import sys
sys.path.insert(0,'.')
sys.path.insert(0,'..')
import torch
import models 
import time
import onnxruntime
import os
import numpy as np
from torch.autograd.variable import Variable
import roc
from sklearn.metrics import confusion_matrix
from tools.data.demo_onnx import surf_demo_single_modal

# def softmax(x):
#     return(np.exp(x)/np.exp(x).sum())

def softmax(x, axis=0):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=axis) # only difference    

if __name__=='__main__':
    name = 'FeatherNetB'
    output_file = f'work_dirs/{name}/{name}.onnx'
    model_path = f'./checkpoints/FeathernetB_bs32/_146_best.pth.tar'


    # name = 'FeatherNetA'
    # output_file = f'work_dirs/{name}/{name}.onnx'
    # model_path = f'./checkpoints/{name}_bs32/_199.pth.tar'

    # name = 'moilenetv2'
    # output_file = f'work_dirs/{name}/{name}.onnx'
    # model_path = f'./checkpoints/mobilenetv2_bs32/_146.pth.tar'

    # name = 'fishnet150'
    # output_file = f'work_dirs/{name}/{name}.onnx'
    # model_path = f'./checkpoints/pre-trainedModels/fishnet150_ckpt.tar'

    net = models.__dict__[name]()
    #print(net)
    
    checkpoint = torch.load(model_path,map_location = 'cpu')
    print('load model:',model_path)
    model_dict = {}
    state_dict = net.state_dict()
    #print(checkpoint)
    for (k,v) in checkpoint['state_dict'].items():
        # print(k)
        if k[7:] in state_dict:
            model_dict[k[7:]] = v
    state_dict.update(model_dict)
    net.load_state_dict(state_dict)

    net.eval()
    dummy_input = torch.randn([1,3,256,256])

    # output_file = f'work_dirs/{name}/feathernetB.onnx'
    if not os.path.exists(output_file):
        torch.onnx.export(net,dummy_input,output_file,verbose=True)


    """ONNX Inference"""
    dummy_input = dummy_input.cpu().numpy()   

    session = onnxruntime.InferenceSession(output_file, None)
    input1_name = session.get_inputs()[0].name
    # input2_name = session.get_inputs()[1].name
    # input3_name = session.get_inputs()[1].name
    st = time.time()
    net_outs = session.run(None, 
            {                
                input1_name: dummy_input,
                # input2_name: inputs_ir[:,frame_t,:,:,:],
                # input3_name: inputs_depth[:,frame_t,:,:,:]
            },)
    output = net_outs[0]   
    soft_output = softmax(output, 1) # torch.Size([32, 1024])
    # prob = np.max(soft_output, 1)[0]
    prob = soft_output[:,1][0]
    predicted = np.argmax(soft_output, 1)[0] 
    print(f"predicted:{predicted} prob: {prob*100}%")
    latency = (time.time()-st) * 1000
    print(f"latency: {latency} ms")  

    #################################################################

    from tools.data.Loadtemporal_valtest_3modality import *     
    image_dir = '/home/leyan/DataSet/CASIA-CeFA/phase2'    
    test_list = f"{image_dir}/test_res.txt"
    val_dataset = Spoofing_valtest(test_list, image_dir, transform=transforms.Compose([
                                    Normaliztion_valtest(), ToTensor_valtest()]))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
    label_list, predicted_list, result_list = [], [], []

    time_list = []

    with torch.no_grad():
        for i, (sample_batched) in enumerate(val_loader):
            with torch.no_grad():
                input = sample_batched['image_depth'].squeeze(1).numpy()
                target = sample_batched['live'].numpy()[0]
                
                st = time.time()
                net_outs = session.run(None, 
                {                
                    input1_name: input,                    
                },)
                output = net_outs[0]   
                soft_output = softmax(output, 1) # torch.Size([32, 1024])
                # prob = np.max(soft_output, 1)[0]
                prob = soft_output[:,1][0]
                predicted = np.argmax(soft_output, 1)[0] 
                latency = (time.time()-st) * 1000
                text = f"latency: {latency} ms, real:{target} predicted:{predicted} real_prob: {prob*100}%"
                if target != predicted: text = text + "\n predict wrong"
                print(text)
                time_list.append(latency)

                label_list.append(target)
                predicted_list.append(predicted)
                result_list.append(soft_output[:,1][0])

    avg_time = sum(time_list) / len(time_list)
    print(f"average latency time:{avg_time} ms")

    tn, fp, fn, tp = confusion_matrix(label_list, predicted_list).ravel()
    apcer = fp/(tn + fp)
    npcer = fn/(fn + tp) # BPCER
    acer = (apcer + npcer)/2
    metric =roc.cal_metric(label_list, result_list)
    eer = metric[0]
    tprs = metric[1]
    auc = metric[2]
    acc = ((tp + tn) / len(label_list))*100

    result_line = 'EER: {:.6f} TPR@FPR=10E-2: {:.6f} TPR@FPR=10E-3: {:.6f} \n \
    APCER:{:.6f} NPCER:{:.6f} AUC: {:.8f} Acc:{:.3f} \n \
    TN:{} FP : {} FN:{} TP:{} \n \
    ACER:{:.8f} '.format(eer, tprs["TPR@FPR=10E-2"], tprs["TPR@FPR=10E-3"],apcer,npcer,auc, acc, tn, fp, fn,tp,acer)
    print(result_line)


    """SURF-DEMO"""
    # surf_demo_single_modal(scrfd_file="./checkpoints/scrfd_2.5g_shape256x256.onnx",
    #     mcdcn_file=output_file)
