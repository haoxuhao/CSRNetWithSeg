#!/usr/bin/env python

import random
import os
from PIL import Image,ImageFilter,ImageDraw
import numpy as np
import h5py
from PIL import ImageStat
import cv2
from tools.utils import get_gt_filepath

def load_data(img_path,train = True, unlabel=False, resize=False, width=1024, height=768, stride = 8, \
        random_crop=False, flip_lr=True, crop_scale=1/2):

    if unlabel:
        img = Image.open(img_path).convert('RGB')
        if resize:
            img = img.resize((width,height))
        img_width = img.size[0]
        img_height = img.size[1]
        unlabel_img = img.resize((img_width//stride, img_height//stride))

        return np.float32(img), unlabel_img
    gt_path = os.path.join("/home/xuhao/dataset/shanghai/SHA/geo_ada_bound_train/", os.path.basename(img_path).replace(".jpg",".h5"))
    #gt_path = get_gt_filepath(img_path, mode=".h5", dataset="SH")
    
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density'])
    if resize:
        img=img.resize((width,height))
    
        img_width=img.size[0]
        img_height=img.size[1]

        scale=target.shape[1]/img_width
        scale*=target.shape[0]/img_height
        target=cv2.resize(target, (img_width, img_height),interpolation = cv2.INTER_CUBIC)*scale

    if random_crop and train:
        crop_size = (int(img.size[0]*crop_scale),int(img.size[1]*crop_scale))
        if random.randint(0,9)<= -1:
            
            dx = int(random.randint(0,1)*img.size[0]*(1-crop_scale))
            dy = int(random.randint(0,1)*img.size[1]*(1-crop_scale))
        else:
            dx = int(random.random()*img.size[0]*(1-crop_scale))
            dy = int(random.random()*img.size[1]*(1-crop_scale))
        
        img = img.crop((dx,dy,crop_size[0]+dx,crop_size[1]+dy))
        target = target[dy:crop_size[1]+dy,dx:crop_size[0]+dx]
        
        
    if flip_lr and train: #flip left to right
        if random.random()>0.6:
            target = np.fliplr(target)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

    target = cv2.resize(target,(int(target.shape[1]/stride),int(target.shape[0]/stride)),interpolation = cv2.INTER_CUBIC)*stride*stride
    target=np.reshape(target,(1,target.shape[0],target.shape[1]))
    target=np.float32(target)

    return img,target

if __name__=="__main__":
    import json
    image_paths=json.load(open('./dataset/SH_partB_train.json','r'))
    for img_path in image_paths[0:20]:
        img, target=load_data(img_path,train = True, random_crop=True)
        print(img.size)
        print(target.shape)
