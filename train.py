#!/usr/bin/env python
#encoding=utf8

import os
import random
import torch
import torch.nn as nn
import numpy as np
import argparse
import json
import cv2
import dataset
import time
from torch.autograd import Variable
from torchvision import datasets, transforms
from model import CSRNetWithSeg
from utils import save_checkpoint


parser = argparse.ArgumentParser(description='PyTorch CSRNetWithSeg')

parser.add_argument('train_json', metavar='TRAIN',
                    help='path to train json')
parser.add_argument('--val_json', metavar='TEST',default=None,type=str,
                    help='path to test json')

parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None,type=str,
                    help='path to the pretrained model')

parser.add_argument('gpu',metavar='GPU', type=str,
                    help='GPU id to use.')

parser.add_argument('task',metavar='TASK', type=str,
                    help='task id to use.')


backup_dir_root = "./backup"

def main():
    global args,best_prec1
    global log_file
    best_prec1 = 1e6
    
    args = parser.parse_args()
    args.lambd = 1.0 # balance segmentation loss and regression loss
    args.lr = 1e-5
    args.batch_size    = 1
    args.momentum      = 0.95
    args.decay         = 5*1e-4
    args.start_epoch   = 0
    args.epochs = 550
    args.steps         = [-1,1,100,150]
    args.scales        = [1,1,1,1]
    args.workers = 8
    args.seed = time.time()
    args.print_freq = 5
    args.loss_mult = 1e1

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.cuda.manual_seed(args.seed)

    backupdir = os.path.join(backup_dir_root, args.task)
    if not os.path.exists(backupdir):
        os.makedirs(backupdir)

    log_file=open("%s/%s_train.log"%(backupdir, args.task),"a")
    val_log_file=open("%s/%s_val_best.log"%(backupdir, args.task),"a")

    with open(args.train_json, 'r') as outfile:        
        train_list = json.load(outfile)
    if args.val_json==None:
        val_list=train_list[0:int(0.1*len(train_list))]
        train_list=train_list[int(0.1*len(train_list)):len(train_list)]
        random.shuffle(train_list)
    else:
        with open(args.val_json, 'r') as outfile:       
            val_list = json.load(outfile)
    print("usage gpu:%s"%args.gpu)
    print("train set size:%d"%(len(train_list)))
    print("val set size:%d"%(len(val_list)))
    
    # model define
    model = CSRNetWithSeg(deformable=False, BN=False, 
                          with_seg = True, shallow=False)

    # size of feature map = 1/stride * image_size 
    args.stride = model.stride 
    model = model.cuda()
    regloss = nn.MSELoss(size_average=False).cuda()
    segloss = nn.BCELoss(size_average=False).cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.decay)

    if args.pre:
        if os.path.isfile(args.pre):
            print("=> loading checkpoint '{}'".format(args.pre))
            checkpoint = torch.load(args.pre)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' epoch {}, best MAE {mae:.3f}"
                  .format(args.pre, checkpoint['epoch'],mae=best_prec1))
        else:
            print("=> no checkpoint found at '{}'".format(args.pre))
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        train(train_list, model, regloss, segloss, optimizer, epoch)
        prec1 = validate(val_list, model, regloss)
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)

        print(' * best MAE {mae:.3f} '
              .format(mae=best_prec1))

        val_log_file.write("epoch:%d\tmae:%.3f\tbest_mae:%.3f\n"%(epoch+1, prec1,best_prec1))
        val_log_file.flush()
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.pre,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best,backupdir)
    log_file.close()

def train(train_list, model, regloss, segloss, optimizer, epoch):
    losses = AverageMeter()
    seg_losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(train_list,
                       shuffle=True,
                       transform=transforms.Compose([
                            transforms.ColorJitter(brightness=0.8, \
                                    contrast=0.9, saturation=0.9), #color agumentation
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ]), 
                       train=True, 
                       seen=model.seen,
                       batch_size=args.batch_size,
                       num_workers=args.workers,
                       stride = args.stride),
        batch_size=args.batch_size)

    print('epoch %d, processed %d samples, lr %s' % (epoch, epoch * len(train_loader.dataset), format(args.lr,'.2e')))
    
    model.train()
    end = time.time()
    
    for i,(img, target,_)in enumerate(train_loader):
        data_time.update(time.time() - end)
        optimizer.zero_grad()
        img = img.cuda()
        img = Variable(img)
        if model.with_seg:
            output, seg_out = model(img)
            bin_seg_out = torch.where(seg_out > 0.5,torch.full_like(seg_out, 1),torch.full_like(seg_out, 0))
            output = output*bin_seg_out
        else:
            output = model(img)
        target = target.type(torch.FloatTensor).cuda()
        target = Variable(target)
        loss = regloss(output, target) #target shape is [1,batch,w,h]
        
        # update moving average
        losses.update(loss.item(), img.size(0))
        
        if model.with_seg:
            front_target = torch.where(target > 1e-6, torch.full_like(target, 1), torch.full_like(target, 0))
            seg_target = Variable(front_target.type(torch.FloatTensor).cuda())
            seg_loss = segloss(seg_out, seg_target)
            loss += args.lambd * seg_loss
            seg_losses.update(seg_loss.item(), img.size(0))
        
        loss.backward(retain_graph=True)
        optimizer.step()    
        
        #output messages
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            if model.with_seg:
                print('Epoch: [{0}/{1}][{2}/{3}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'SegLoss {seg_losses.val:.4f} ({seg_losses.avg:.4f})\t'
                    .format(
                    epoch, args.epochs, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, seg_losses=seg_losses))
            else:
                print('Epoch: [{0}/{1}][{2}/{3}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    .format(
                    epoch, args.epochs, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses))

    
def validate(val_list, model):
    print ('begin val')
    test_loader = torch.utils.data.DataLoader(
    dataset.listDataset(val_list,
                   shuffle=False,stride=args.stride,
                   transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ]),  train=False),
    batch_size=1)    
    model.eval()
    mae = 0
    hit_rate_sum = 0
    for i,(img, target,_) in enumerate(test_loader):
        img = img.cuda()
        target = target.cuda()
        img = Variable(img)
        if model.with_seg:
            output, seg_out = model(img)
            front_target = torch.where(target > 1e-6, torch.full_like(target, 1), torch.full_like(target, 0))
            seg_res = torch.where(seg_out > 0.35, torch.full_like(seg_out, 1), torch.full_like(seg_out, 0))

            seg_hit = torch.where(front_target==seg_res, torch.full_like(seg_res, 1), torch.full_like(seg_res, 0))
            front_target_shape = front_target.cpu().numpy().shape
            total_sum = front_target_shape[2]*front_target_shape[3]
            hit_rate = seg_hit.cpu().numpy().sum()/total_sum
            hit_rate_sum+=hit_rate
            
            output = output*seg_res
        else:
            output = model(img)
        
        mae += abs(output.data.sum()-target.sum().type(torch.FloatTensor).cuda())

    hit_rate_avg = float(hit_rate_sum)/len(test_loader)
    print("* hit rate: %.4f"%(hit_rate_avg))

    mae = mae/len(test_loader)    
    print(' * MAE {mae:.3f} '
              .format(mae=mae))

    return mae    
        
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    #read cfg from cfg file
    with open("cfg.json",'r') as f:
        train_cfg=json.load(f)[0]["train"]
        args.original_lr=train_cfg["original_lr"]
        args.scales = train_cfg["scales"]
        args.steps = train_cfg["steps"]
        args.lambd = train_cfg["lambd"]
        args.epochs = train_cfg["epochs"]

    args.steps.append(args.epochs)

    args.lr = args.original_lr
    scale = 1
    for i, step in enumerate(args.steps[:len(args.steps)-1]):       
        if epoch >= step and epoch < args.steps[i+1]:
            scale = args.scales[i] if i < len(args.scales) else args.scales[-1]
            break
    args.lr = args.original_lr*scale
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr
        
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count    
    
if __name__ == '__main__':
    main()        
