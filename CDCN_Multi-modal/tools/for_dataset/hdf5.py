import h5py
import cv2
from glob import glob
import os
import numpy as np

fns = '/home/leyan/DataSet/HQ-WMCA_MCCNN-128/HQ-WMCA/MCCNN-128/preprocessed/face-station/*/*.hdf5'
stem = "/home/leyan/DataSet/HQ-WMCA_MCCNN-128/HQ-WMCA/MCCNN-128/preprocessed/face-station/"

# MCCNN-128: This is the set of preprocessed files to use with the MCCNN models
# presented in the reference paper. The images are of size (128 × 128) and the face crop is
# loose made suitable for the LightCNN model

# Each saved file has the following name format:
# <site_id>_<session_id>_<client_id>_<presenter_id>_<type_id>_<sub_type_id>_<pai_id>.hdf5
# ex: 01.03.19/1_01_0001_0000_00_00_000-241b2419.hdf5

"""
• site_id: The number represents the place of the data collection. In this database this number
is always ‘1’.

• session_id: The number associated to a session as mentioned before.

• client_id: This number presents the identity of what is presented to the system. For bonafide,
it is the ID given to the participant upon arrival and for the attacks, it is a number given to a
PAI in this protocol. Please note that if the identity of a subject is the same as the identity
of an attack this number is the same for both cases. One example is the silicon masks. If a
silicon mask is made from subject ‘x’ and subject ‘x’ also participated as bonafide in the
data collection the “client_id” for bonafide and silicon mask is the same.

• presenter_id: If a subject is presenting an attack to the system, this number is the subject’s
“client_id”. If the attack is presented on a support, this number is ‘0000’. If the capture is for
bonafide this number is ‘0000’ as well since there is no presenter in this case.

• type_id: The attack types mentioned in 4.2. For bonafide this number is ‘00’.

• sub_type_id: The sub_types for each attack type mentioned in 4.2. For bonafide without
glasses this number is ‘00’ and if they wore medical glasses this number is ’01’.

• pai_ id: The unique number associated with each and every PAI. This number for bonafide
both with and without medical glasses is ‘000’.
"""

for fn in glob(fns):
    fn = fn.replace(stem, "")
    dir_name = os.path.dirname(fn)
    # print(dir_name)
    print(fn)

    h5_file = h5py.File(os.path.join(stem, fn), 'r')

    for keys in h5_file.keys():
        # print("--------------------------")
        # print(keys)
        dataset = h5_file.get(keys)
        for key in list(dataset.keys()):
            if key == "array":
                # continue
                print(dataset[key])
                arr = dataset[key][:] 
                
                # total of 43 channels, one grayscale image and the 42 combinations of SWIR wavelengths
                print(type(arr))
                img_nums = len(arr)   

                # for i in range(img_nums):          
                #     img = cv2.cvtColor(arr[i], cv2.COLOR_RGB2BGR)
                #     cv2.imshow("w", img)
                #     c= cv2.waitKey(1) & 0xff
                gray_image = arr[0]
                
                rgb_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)

                # Duplicate the gray values in each channel
                rgb_image[:,:,0] = gray_image
                rgb_image[:,:,1] = gray_image
                rgb_image[:,:,2] = gray_image


                # rgb = np.array([arr[0]]*3)
                # rgb = np.transpose(rgb, (1, 2, 0))
                # img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.imshow("w", rgb_image)
                c= cv2.waitKey(1) & 0xff
            # else:
            #     print(dataset[key])
            #     print(dataset[key].name)
    # break


exit(0)
fn = '/home/leyan/DataSet/HQ-WMCA_MCCNN-128/HQ-WMCA/MCCNN-128/preprocessed/face-station/01.03.19/1_01_0001_0000_00_00_000-241b2419.hdf5'
h5_file = h5py.File(fn, 'r')

# dataset = h5_file.get('FrameIndexes')
# data = dataset.value

for keys in h5_file.keys():
    print("--------------------------")
    # print(keys)
    dataset = h5_file.get(keys)
    for key in list(dataset.keys()):
        if key == "array":
            print(dataset[key])
            arr = dataset[key][:] 
            print(type(arr))
            img_nums = len(arr)   

            for i in range(img_nums):          
                img = cv2.cvtColor(arr[i], cv2.COLOR_RGB2BGR)
                cv2.imshow("w", img)
                c= cv2.waitKey(1) & 0xff

                # if c==27:
                #     break


    # print(dataset.values())
    # print(list(dataset.keys()))
    # print(dataset["array"])
    # print(h5_file[key].name)
    # print(h5_file[key].shape)
    # print(h5_file[key].value)


import pandas as pd
# import tables
# 将mode改成r即可
hdf5 = pd.HDFStore(fn, mode="r")
# print(hdf5.keys())
hdf5.walk()

pass