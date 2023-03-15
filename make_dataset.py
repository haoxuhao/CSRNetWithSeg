#!/usr/bin/env python
# coding: utf-8

import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter 
import scipy
import json
from matplotlib import cm as CM
from image import *
from tools.draw import *
import threading

"""
Generate density map as groundtruth
"""

#this is borrowed from https://github.com/davideverona/deep-crowd-counting_crowdnet
def gaussian_filter_density(gt):
    #print(gt.shape)
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density
    #print(gt_count) 
    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    leafsize = 2048
    
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)

    #print('generate density...')
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1],pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        else:
            sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point

        sigma = 20 if sigma > 20 else sigma
        sigma = 5 if sigma < 5 else sigma

        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    #print('done.')
    return density

den_map_save_tmp = "/home/xuhao/temp/SHB_train_ada_gen"
if not os.path.exists(den_map_save_tmp):
    os.makedirs(den_map_save_tmp)

threadLock = threading.Lock()
counter = 0
class WorkThread (threading.Thread):
    def __init__(self, threadID, task, total, use_geo=True):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.task = task
        self.use_geo = use_geo
        self.total=total

    def run(self):
        for img_path in task: 
            #print("[%d]: %s start."%(self.threadID, img_path))
            if not os.path.exists(img_path):
                print("[%d]: %s no such file."%(self.threadID, img_path))
                continue

            mat = io.loadmat(img_path.replace('.jpg','.mat').replace('images','ground_truth').replace('IMG','GT_IMG'))

            img= plt.imread(img_path)
            k = np.zeros((img.shape[0],img.shape[1]))
            gt = mat["image_info"][0,0][0,0][0]
            #gt = mat["image_info"]
            
            for i in range(0,len(gt)):
                if int(gt[i][1])<img.shape[0] and int(gt[i][0])<img.shape[1]:
                    k[int(gt[i][1]),int(gt[i][0])]=1
            if self.use_geo:
                k = gaussian_filter_density(k)

                den_file_name = os.path.basename(img_path).replace('.jpg','.h5').replace('images','ground_truth')
                
                with h5py.File(os.path.join(den_map_save_tmp, den_file_name), 'w') as hf:
                    hf['density'] = k 

                # with h5py.File(img_path.replace('.jpg','.h5').replace('images','ground_truth'), 'w') as hf:
                #         hf['density'] = k
            else:
                k = gaussian_filter(k,15) #fixed gaussian kernal

                den_file_name = os.path.basename(img_path).replace('.jpg','.h5').replace('images','ground_truth')
                
                with h5py.File(os.path.join(den_map_save_tmp, den_file_name), 'w') as hf:
                    hf['density'] = k 

                # with h5py.File(img_path.replace('.jpg','.h5').replace('images','ground_truth'), 'w') as hf:
                #     hf['density'] = k

            threadLock.acquire()
            global counter
            counter+=1
            threadLock.release()

            print("[%d/%d] %s done."%(counter, self.total, img_path))
        print("thread %d exit."%self.threadID)


def draw_dens(im, save_path, den1=None, den2=None, den3=None, index=0):
    plt.figure(index,figsize=(10,8))
    plt.subplot(221)
    plt.title('image')
    plt.imshow(im, alpha=1)

    if den1 is not None:
        plt.subplot(222)
        plt.title('den1')
        plt.imshow(den1,cmap=CM.jet,alpha=1)
    if den2 is not None:
        plt.subplot(223)
        plt.title('den2')
        plt.imshow(den2,cmap=CM.jet,alpha=1)
    if den3 is not None:
        plt.subplot(224)
        plt.title('den3')
        plt.imshow(den3,cmap=CM.jet,alpha=1)
    plt.savefig(save_path)
    plt.close()

def show_gts(img_list):
    save_path="/home/xuhao/temp/"
    for file in img_list:
        print(file)
        img_filename = os.path.basename(file)
        gt_file_path = os.path.join(den_map_save_tmp, img_filename.replace(".jpg",".h5"))
        save_file_path = os.path.join(save_path,"den_map_"+img_filename)
        img=Image.open(file).convert('RGBA')

        gt_file = h5py.File(gt_file_path,'r')
        den1 = np.asarray(gt_file['density'])
        gt_file = h5py.File(file.replace('.jpg','.h5').replace('images','ground_truth'),'r')
        den2 = np.asarray(gt_file['density'])
        
        draw_dens(img, save_file_path, den1 = den1, den2 = den2)

gen_den = True
if gen_den:
    # #set the root to the Shanghai dataset you download
    SHA_train_set = json.load(open("dataset/SH_partB_train.json"))
    SHA_test_set = json.load(open("dataset/SH_partB_test.json"))

    sum_set = SHA_train_set

    # sum_set=train_set+test_set
    # print("train set: %d"%len(train_set))
    # print("test set: %d"%len(test_set))
    print("sum set: %d"%len(sum_set))
    # with open(os.path.join(den_map_save_tmp, "images.txt"), 'w') as f:
    #     json.dump(sum_set, f)

    thread_num=4

    if len(sum_set)<thread_num:
        thread_num=1

    stride=int(len(sum_set)/thread_num)
    start=0

    total=len(sum_set)

    workers=[]
    for i in range(thread_num):
        if i==(thread_num-1):
            task=sum_set[start: len(sum_set)]
            start=len(sum_set)
        else:
            task=sum_set[start:(i+1)*stride]
            start=(i+1)*stride

        worker=WorkThread(i,task,total)
        #worker=WorkProcess(i,task,total)
        worker.start()
        workers.append(worker)

    for worker in workers:
        worker.join()

else:
    with open(os.path.join(den_map_save_tmp, "images.txt")) as f:
        sum_set = json.load(f)
    show_gts(sum_set)



