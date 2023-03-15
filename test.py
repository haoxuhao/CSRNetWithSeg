#!/usr/bin/env python
# coding: utf-8

import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter 
import scipy
import json
import torchvision.transforms.functional as F
from matplotlib import cm as CM
from model import CSRNetWithSeg
import torch
import time
import json
from torchvision import datasets, transforms
import math
import sys
from tools import utils
from tools.utils import get_gt_filepath
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ])

test_set="./dataset/SH_partB_test.json"
img_paths=json.load(open(test_set,'r'))
print("test set length:%d"%len(img_paths))

model_path = "./backup/semi_gan_test_multi_task/G_best.pth.tar"
model_path = "backup/semi_gan_test_multi_task/G_best.pth.tar"
model_path = "models/partBmodel_best.pth.tar"

model_path = "./backup/SHB_csrnet_with_seg_whole/best.pth.tar"
model = CSRNetWithSeg(BN=False, deformable=False, with_seg=True, shallow=False)
model = model.cuda()
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['state_dict'])
mae_rate_sum = 0
mae_sum = 0
sum_time=0
mse_sum=0
detail_results=[]
seg_map_root = "results/temp/SHA_seg_map"
with_mask = False

for i in range(len(img_paths)):#
    img=Image.open(img_paths[i])
    
    #resize 1.5
    new_w, new_h = int(img.size[0]), int(img.size[1])
    img = img.resize((new_w,new_h))

    #new_w,new_h = img.size

    img = transform(img.convert('RGB')).cuda()

    gt_path = get_gt_filepath(img_paths[i], mode=".mat",dataset="SH")
    # gt_file = h5py.File(gt_path,'r') #use h5
    # groundtruth = np.asarray(gt_file['density'])
    # gt_count = groundtruth.sum()
   
    mat = io.loadmat(gt_path) #use .mat
    gt_count = len(mat["image_info"][0,0][0,0][0])

    start=time.time()
    if model.with_seg:
        output, seg_out = model(img.unsqueeze(0))
        thresh = 0.5   
        seg_res = torch.where(seg_out > thresh, torch.full_like(seg_out, 1), torch.full_like(seg_out, 0))
        output = seg_res*output
        pred_den = (output.detach().cpu().numpy())
        pred_count = pred_den.sum()
        
    else:
        output = model(img.unsqueeze(0))
        density_map = output.detach().cpu().numpy()[0,0]
        
        density_map = cv2.resize(density_map/64,(int(density_map.shape[1]*8),int(density_map.shape[0]*8)),\
            interpolation = cv2.INTER_CUBIC)

        if with_mask:
            #load mask
            mask = np.load(os.path.join(seg_map_root, os.path.basename(img_paths[i].replace(".jpg", ".npy"))))
            thresh = 0.1
            mask = np.where(mask >=thresh, 1, 0)
            density_map = density_map*mask

        pred_count=density_map.sum()
        

    sum_time+=time.time()-start

    mae=abs(pred_count-gt_count)
    mse_sum+=(mae*mae)
    mae_rate=mae/gt_count
    mae_rate_sum+=mae_rate
    mae_sum += mae
    
    #build the detail results
    res={}
    res["path"]=img_paths[i]
    res["den_path"]=gt_path
    res["gt_count"]=float(gt_count)
    res["pred_count"]=float(pred_count)
    detail_results.append(res)

    print("[%d/%d] %s err: %.4f"%(i+1,len(img_paths),img_paths[i],mae))
    
print("model:%s"%model_path)
print("test set:%s"%test_set)
print("================")
print("mae: %.4f"%(mae_sum/len(img_paths)))
print("mae rate: %.4f"%(mae_rate_sum/len(img_paths)))
print("mse: %.4f"%(math.sqrt(mse_sum/len(img_paths))))
print("average inference time: %.4f"%(sum_time/len(img_paths)))

res_file="results/partb/val_detail_res_seg_partb.json"

with open(res_file,"w") as file:
    json.dump(detail_results, file, sort_keys=True, indent=4, separators=(',', ': '))

