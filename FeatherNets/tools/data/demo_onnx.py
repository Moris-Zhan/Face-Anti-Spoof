import onnxruntime
from glob import glob
import os
import cv2
import numpy as np
import time

def softmax(x, axis=0):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=axis) # only difference    

def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)

def nms(dets):
    thresh = nms_thresh = 0.4
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def pred_scrfd_bbox(img, thresh=0.5, input_size = None, max_num=0, metric='default', session=None):
    st = time.time()
    """prepare Img"""
    # input_size = (256, 256)
    im_ratio = float(img.shape[0]) / img.shape[1]
    model_ratio = float(input_size[1]) / input_size[0]
    if im_ratio>model_ratio:
        new_height = input_size[1]
        new_width = int(new_height / im_ratio)
    else:
        new_width = input_size[0]
        new_height = int(new_width * im_ratio)
    det_scale = float(new_height) / img.shape[0]
    resized_img = cv2.resize(img, (new_width, new_height))
    det_img = np.zeros( (input_size[1], input_size[0], 3), dtype=np.uint8 )
    det_img[:new_height, :new_width, :] = resized_img

    """forward"""
    scores_list = []
    bboxes_list = []
    kpss_list = []
    input_size = tuple(det_img.shape[0:2][::-1])
    blob = cv2.dnn.blobFromImage(det_img, 1.0/128, input_size, (127.5, 127.5, 127.5), swapRB=True)
    # session = onnxruntime.InferenceSession(model_file, None)
    net_outs = session.run(None, 
            {"input.1": blob
             },)

    input_height = blob.shape[2]
    input_width = blob.shape[3]
    fmc = 3
    _feat_stride_fpn = [8, 16, 32]
    _num_anchors = 2

    """decode"""
    for idx, stride in enumerate(_feat_stride_fpn):
        # If model support batch dim, take first output
        scores = net_outs[idx][0]
        # savetxt('scores_%s.csv'%(stride), scores, delimiter=',', fmt='% f')
        bbox_preds = net_outs[idx + fmc][0]
        # savetxt('bbox_preds_%s.csv'%(stride), bbox_preds, delimiter=',', fmt='% f')
        bbox_preds = bbox_preds * stride
        
        height = input_height // stride
        width  = input_width // stride

        # print("height: ", height)
        # print("width: ", width)
        K = height * width
        key = (height, width, stride)

        #solution-1, c style:
        anchor_centers = np.zeros( (height, width, 2), dtype=np.float32 )
        for i in range(height):
            anchor_centers[i, :, 1] = i
        for i in range(width):
            anchor_centers[:, i, 0] = i

        anchor_centers = (anchor_centers * stride).reshape( (-1, 2) )
        # savetxt('anchor_centers_%s.csv'%(stride), anchor_centers, delimiter=',', fmt='% 4d')

        if _num_anchors>1:
            anchor_centers = np.stack([anchor_centers]*_num_anchors, axis=1).reshape( (-1,2) )

        pos_inds = np.where(scores>=thresh)[0]
        bboxes = distance2bbox(anchor_centers, bbox_preds) # box anchor to coord [x1,y1,x2,y2]
        pos_scores = scores[pos_inds]
        pos_bboxes = bboxes[pos_inds]
        scores_list.append(pos_scores)
        bboxes_list.append(pos_bboxes)   

    """post process"""
    scores = np.vstack(scores_list)
    scores_ravel = scores.ravel()
    order = scores_ravel.argsort()[::-1]

    bboxes = np.vstack(bboxes_list) / det_scale
    # print("img.shape[0]: ", img.shape[0])
    # print("new_height: ", new_height)
    # print("det_scale: ", det_scale)
    
    pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
    pre_det = pre_det[order, :]
    keep = nms(pre_det)
    det = pre_det[keep, :]

    if max_num > 0 and det.shape[0] > max_num:
        area = (det[:, 2] - det[:, 0]) * (det[:, 3] -
                                                det[:, 1])
        img_center = img.shape[0] // 2, img.shape[1] // 2
        offsets = np.vstack([
            (det[:, 0] + det[:, 2]) / 2 - img_center[1],
            (det[:, 1] + det[:, 3]) / 2 - img_center[0]
        ])
        offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
        if metric=='max':
            values = area
        else:
            values = area - offset_dist_squared * 2.0  # some extra weight on the centering
        bindex = np.argsort(
            values)[::-1]  # some extra weight on the centering
        bindex = bindex[0:max_num]
        det = det[bindex, :]
    latency = time.time()-st
    fps  = (1/latency)
    return det, latency, fps     

def pad_img(crop_img, new_width, new_height, input_size):
    resized_img = cv2.resize(crop_img, (new_width, new_height))
    det_img = np.zeros( (input_size[1], input_size[0], 3), dtype=np.uint8 )
    det_img[:new_height, :new_width, :] = resized_img
    return det_img

def recogn_spoof_mcdcn(rgb, depth, ir, bboxes, session=None):
    st = time.time()
    """Pre-Process"""
    x1,y1,x2,y2,score = bboxes[0].astype(np.int)
    crop_rgb = rgb[y1:y2, x1:x2]
    crop_depth = depth[y1:y2, x1:x2]
    crop_ir = ir[y1:y2, x1:x2]
    # input_size = (crop_ir.shape[1], crop_ir.shape[0])

    input_size = (256, 256)
            
    im_ratio = float(crop_ir.shape[0]) / crop_ir.shape[1]
    model_ratio = float(input_size[1]) / input_size[0]
    if im_ratio>model_ratio:
        new_height = input_size[1]
        new_width = int(new_height / im_ratio)
    else:
        new_width = input_size[0]
        new_height = int(new_width * im_ratio)
    # det_scale = float(new_height) / crop_ir.shape[0]
    # resized_img = cv2.resize(crop_rgb, (new_width, new_height))
    # det_img = np.zeros( (input_size[1], input_size[0], 3), dtype=np.uint8 )

    pad_rgb = pad_img(crop_rgb, new_width, new_height, input_size)
    pad_ir = pad_img(crop_ir, new_width, new_height, input_size)
    pad_depth = pad_img(crop_depth, new_width, new_height, input_size)

    blob_rgb = cv2.dnn.blobFromImage(pad_rgb, 1.0/128, input_size, (127.5, 127.5, 127.5), swapRB=True)
    blob_depth = cv2.dnn.blobFromImage(pad_depth, 1.0/128, input_size, (127.5, 127.5, 127.5), swapRB=True)
    blob_ir = cv2.dnn.blobFromImage(pad_ir, 1.0/128, input_size, (127.5, 127.5, 127.5), swapRB=True)

    input1_name = session.get_inputs()[0].name
    # input2_name = session.get_inputs()[1].name
    # input3_name = session.get_inputs()[2].name

    # st = time.time()
    net_outs = session.run(None, 
            {
                input1_name: blob_depth,
                # input2_name: blob_ir,
                # input3_name: blob_depth
            },)
    # print(f"Duration Time:  {time.time()-st:2.2f}, s")
    output = net_outs[0]   
    soft_output = softmax(output, 1) # torch.Size([32, 1024])
    # prob = np.max(soft_output, 1)[0]
    predicted = np.argmax(soft_output, 1)[0]
    prob = soft_output[:,1][0]
    # map_x = net_outs[0]
    # score_norm = np.sum(map_x)/np.sum(pad_depth)
    # print("score_norm: ", score_norm)

    # crop_img = np.hstack([crop_rgb, crop_ir, crop_depth, ])
    # cv2.imshow("", crop_img)
    # cv2.waitKey(0)
    latency = time.time()-st
    fps  = (1/latency)
    text = f"latency: {latency} ms, predicted:{predicted} real_prob: {prob*100}%"
    print(text)
    return prob, latency, fps   

"""Real-People"""
def surf_demo_single_modal(
        scrfd_file="./tools/scrfd_2.5g_shape256x256.onnx",
        mcdcn_file="./tools/mcdcn.onnx"):

    directory = "/home/leyan/DataSet/Surfing-2020-Anti-spoofing/data"
    path = directory
    path = os.path.normpath(path)
    res = []
    for root,dirs,files in os.walk(path, topdown=True):
        depth = root[len(path) + len(os.path.sep):].count(os.path.sep)
        if depth == 2:
            # We're currently two directories in, so all subdirs have depth 3
            res += [os.path.join(root, d) for d in dirs]
            dirs[:] = [] # Don't recurse any deeper

    res.sort()

    """Prepare demo_3modal file"""
    video_save_path = "./tools/demo_3modal.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    win_size = (480*3, 640)
    video_fps = 25.0
    out = cv2.VideoWriter(video_save_path, fourcc, video_fps, win_size) 
    
    scrfd_session = onnxruntime.InferenceSession(scrfd_file, None)
    mcdcn_session = onnxruntime.InferenceSession(mcdcn_file, None)

    # dir0 = res
    # print(dir0)
    # count = glob(f"{dir0}/*")
    # print(count)
    directory = "/home/leyan/DataSet/Surfing-2020-Anti-spoofing/data"
    try:
        for dir0 in res:
            # print(dir0)
            folders = glob(f"{dir0}/*", recursive=True)
            folders = [f for f in folders if not os.path.isfile(f)]

            rgb_paths = glob(f"{folders[-1]}/*")
            rgb_paths.sort()
            for rgb_path in rgb_paths:
                # serial = os.path.basename(rgb_path).replace(".png","")  
                serial = os.path.dirname(rgb_path).replace(directory, "")              
                ir_path = rgb_path.replace("rgb","ir").replace("jpg","png")
                depth_path = rgb_path.replace("rgb","depth").replace("jpg","png")
                depth = cv2.imread(depth_path)

                if os.path.exists(ir_path): 
                    pass
                else:
                    ir_path = ir_path.replace("png","jpg")
                    depth = cv2.rotate(depth, cv2.ROTATE_90_COUNTERCLOCKWISE)

                rgb = cv2.imread(rgb_path)
                
                ir = cv2.imread(ir_path)

                bboxes, latency1, _ = pred_scrfd_bbox(rgb, 0.5, input_size = (256, 256), session = scrfd_session)
                # map_score = 0
                prob, latency2, _ = recogn_spoof_mcdcn(rgb, depth, ir, bboxes, session = mcdcn_session)       
                latency = latency1 + latency2    
                fps = 1 / latency     
                print(f"Duration Time:  {latency:2.2f}, s")           
                text_label1 = f"scrfd latency={latency1:2.2f}s, featherNet latency={latency2*1000:2.2f}ms, real_prob:{(prob*100):2.3f}% "
                text_label2 = f"total latency={latency:2.2f}s fps={fps:2.2f}"
                text_label3 = f"{serial}"
                print(text_label1)
                print(text_label2)

                for i in range(bboxes.shape[0]):
                    bbox = bboxes[i]
                    x1,y1,x2,y2,score = bbox.astype(np.int)
                    cv2.rectangle(rgb, (x1,y1)  , (x2,y2) , (255,0,0) , 2)
                    cv2.rectangle(depth, (x1,y1)  , (x2,y2) , (255,0,0) , 2)
                    cv2.rectangle(ir, (x1,y1)  , (x2,y2) , (255,0,0) , 2)
                
                img = np.hstack([rgb, ir, depth, ])
                img = cv2.putText(img, text_label1, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                img = cv2.putText(img, text_label2, (0, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                img = cv2.putText(img, text_label3, (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # RGB+Depth
                # cv2.imshow(os.path.basename(dir0), img)
                fn = "_".join(rgb_path.split("/")[6:])
                os.makedirs("./tools/demo_3modal/", exist_ok=True)
                cv2.imwrite(f"./tools/demo_3modal/{fn}.jpg", img)
                out.write(img)
                cv2.imshow("", img)

                c= cv2.waitKey(1) & 0xff  
                if c==27:
                    break
                # break
            cv2.destroyAllWindows()
    except:
        out.release()
    out.release()