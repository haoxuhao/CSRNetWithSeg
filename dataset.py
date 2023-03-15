import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from image import *
import torchvision.transforms.functional as F


class listDatasetUnlabel(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None,  stride=8, train=False, seen=0, batch_size=1, num_workers=4):
        super(listDatasetUnlabel, self).__init__()

        if train:
            root = root *1
        random.shuffle(root)
        
        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.stride=stride
        
        
    def __len__(self):
        return self.nSamples
    def __getitem__(self, index):
        assert index <= len(self), 'index range error' 
        
        img_path = self.lines[index]
        
        img, unlabel_img = load_data(img_path,self.train, stride = self.stride, unlabel=True, resize=True)
        
        if self.transform is not None:
            img = self.transform(img)
            unlabel_img = self.transform(unlabel_img)

        return img, unlabel_img

class listDataset(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None, stride=8, train=False, seen=0, batch_size=1, num_workers=4):
        if train:
            root = root *1
        random.shuffle(root)
        
        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.stride = stride
        
        
    def __len__(self):
        return self.nSamples
    def __getitem__(self, index):
        assert index <= len(self), 'index range error' 
        
        img_path = self.lines[index]
        
        img,target = load_data(img_path, self.train, stride = self.stride)
        
        #img = 255.0 * F.to_tensor(img)
        
        #img[0,:,:]=img[0,:,:]-92.8207477031
        #img[1,:,:]=img[1,:,:]-95.2757037428
        #img[2,:,:]=img[2,:,:]-104.877445883


        img_width = img.size[0]
        img_height = img.size[1]
        resized = img.resize((img_width//self.stride, img_height//self.stride))

        if self.transform is not None:
            img = self.transform(img)
            resized = self.transform(resized)

        # print("img shape:%s"%(str(img.shape)))
        # print("target shape:%s"%(str(target.shape)))
        return img,target,resized
