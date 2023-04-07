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
import cv2


def softmax(x, axis=0):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=axis) # only difference    

if __name__=='__main__':
    # name = 'FeatherNetB'
    # output_file = f'work_dirs/{name}/{name}.onnx'
    # model_path = f'./checkpoints/{name}_bs32/_146_best.pth.tar'


    # name = 'FeatherNetA'
    # output_file = f'work_dirs/{name}/{name}.onnx'
    # model_path = f'./checkpoints/{name}_bs32/_199.pth.tar'

    # name = 'moilenetv2'
    # output_file = f'work_dirs/{name}/{name}.onnx'
    # model_path = f'./checkpoints/mobilenetv2_bs32/_39.pth.tar'

    # name = 'FeatherNetModal3' # 13ms
    # output_file = f'work_dirs/FeatherNetModal3-feature-fusion/{name}.onnx'
    # model_path = f'./checkpoints/FeatherNet_bs32_modal3/_24_best.pth.tar'

    # name = 'FeatherNetModal3' # 16ms
    # output_file = f'work_dirs/FeatherNetModal3-pipenet-fusion/{name}.onnx'
    # model_path = f'./checkpoints/FeatherNet_bs32_modal3/_24_best.pth.tar'

    name = 'FeatherNetModal3'
    output_file = f'work_dirs/{name}/{name}.onnx'
    model_path = f'./checkpoints/FeatherNet_bs32_modal3/_144.pth.tar'

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
    # dummy_input = torch.randn([1,3,256,256])
    dummy_input = {'rgb': torch.randn([1,3,256,256]),
                    'ir': torch.randn([1,3,256,256]),
                    'depth': torch.randn([1,3,256,256])}

    # output_file = f'work_dirs/{name}/feathernetB.onnx'
    if not os.path.exists(output_file):
        torch.onnx.export(net,dummy_input,output_file,verbose=True)


    """ONNX Inference"""
    dummy_input = torch.randn([1,3,256,256]).numpy()   

    session = onnxruntime.InferenceSession(output_file, None)
    input1_name = session.get_inputs()[0].name
    input2_name = session.get_inputs()[1].name
    input3_name = session.get_inputs()[2].name
    st = time.time()
    net_outs = session.run(None, 
            {                
                input1_name: dummy_input,
                input2_name: dummy_input,
                input3_name: dummy_input
            },)
    output = net_outs[0]   
    soft_output = softmax(output) # torch.Size([32, 1024])
    prob = np.max(soft_output, 1)[0]
    predicted = np.argmax(soft_output, 1)[0] 
    print(f"predicted:{predicted} prob: {prob*100}%")
    latency = (time.time()-st) * 1000
    print(f"latency: {latency} ms")  

    #################################################################
    video_save_path = f"work_dirs/{name}/inference.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    size = (768, 256)
    video_fps = 15.0
    out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

    video_save_path = f"work_dirs/{name}/Live.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    size = (768, 256)
    video_fps = 15.0
    out_live = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

    video_save_path = f"work_dirs/{name}/Spoof.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    size = (768, 256)
    video_fps = 15.0
    out_spoof = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)


    from tools.data.Loadtemporal_valtest_3modality import * 
    # image_dir = '/home/leyan/DataSet/CASIA-CeFA/phase1' 
    # val_list = f'{image_dir}/b_dev.txt'
    image_dir = '/home/leyan/DataSet/CASIA-CeFA/phase2'    
    test_list = f"{image_dir}/test_res.txt"
    val_dataset = Spoofing_valtest(test_list, image_dir, transform=transforms.Compose([
                                    Normaliztion_valtest(), ToTensor_valtest()]))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
    label_list, predicted_list, result_list = [], [], []

    time_list = []
    with torch.no_grad():
        # for i, (input, target,depth_dirs) in enumerate(val_loader):
        for i, (sample_batched) in enumerate(val_loader):
            with torch.no_grad():
                rgb = sample_batched['image_x'].squeeze(1).numpy()
                ir = sample_batched['image_ir'].squeeze(1).numpy()
                depth = sample_batched['image_depth'].squeeze(1).numpy()

                target = sample_batched['live'].numpy()[0]
                # input_var = Variable(input).float().numpy()
                # target_var = Variable(target).long().numpy()
                st = time.time()
                net_outs = session.run(None, 
                {                
                    input1_name: rgb,
                    input2_name: ir,
                    input3_name: depth
                },)
                output = net_outs[0]   
                soft_output = softmax(output, 1) # torch.Size([32, 1024])
                # prob = np.max(soft_output, 1)[0]
                prob = soft_output[:,1][0]
                predicted = np.argmax(soft_output, 1)[0] 
                latency = (time.time()-st) * 1000
                text = f"latency: {latency:.2f} ms, real:{int(target)==1} predicted:{predicted} real_prob: {prob*100:.2f}%"
                if target != predicted: text = text + "\n predict wrong"
                print(text)
                time_list.append(latency)

                label_list.append(target)
                predicted_list.append(predicted)
                result_list.append(soft_output[:,1][0])

                rgb = rgb[0].transpose(1, 2, 0)
                ir = ir[0].transpose(1, 2, 0)
                depth = depth[0].transpose(1, 2, 0)

                img = (np.hstack([
                    rgb, 
                    ir, 
                    depth
                , ])*128+127.5).astype(np.uint8)  
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cv2.putText(img, text, (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imshow("", img)
                out.write(img)
                if int(target)==1:
                    out_live.write(img)
                else:
                    out_spoof.write(img)
                c= cv2.waitKey(1) & 0xff   
                if c==27:
                    break

    avg_time = sum(time_list) / len(time_list)
    print(f"average latency time:{avg_time} ms")
    out.release()

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
