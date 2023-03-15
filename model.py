import torch.nn as nn
import torch
from torchvision import models
from utils import save_net,load_net
from torch_deform_conv.layers import ConvOffset2D
import torch.nn.functional as F

class CSRNetWithSeg(nn.Module):
    def __init__(self, load_weights=False, deformable=False, BN=False, with_seg = False, shallow=True):
        super(CSRNetWithSeg, self).__init__()
        self.seen = 0
        self.with_seg = with_seg
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_shallow = [256, 128, 64]
        if shallow:
            self.backend_feat  = [256, 128, 64] #shallow
        else:
            self.backend_feat  = [512, 512, 512,256,128,64]

        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat,in_channels = 512,dilation = True, deformable=deformable, batch_norm=BN)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        if self.with_seg:
            self.backend_seg_layer = make_layers(self.backend_shallow,in_channels = 512,dilation = True, deformable=deformable, batch_norm=BN)
            self.output_seg_layer = nn.Conv2d(64, 1, kernel_size=1)
        self.stride=8
        if not load_weights:
            mod = models.vgg16(pretrained = True)
            self._initialize_weights()
            for i in range(len(self.frontend.state_dict().items())):
                list(self.frontend.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]


    def forward(self,x):
        front_feats = self.frontend(x)
        x_count = self.backend(front_feats)
        
        if self.with_seg:
            x_seg = self.backend_seg_layer(front_feats)
            out2 = F.sigmoid(self.output_seg_layer(x_seg))
            out1 = self.output_layer(x_count)
            return out1, out2
        else:
            out1 = self.output_layer(x_count)
            return out1

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
                
def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False, deformable=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for i,v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            if deformable and i>=3: #the last 3 layers use deformable 
                offset_feature=ConvOffset2D(in_channels,dilation=d_rate)
                layers+=[offset_feature]
            else:
                pass

            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)                