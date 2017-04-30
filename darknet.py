import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from region_loss import RegionLoss
from utils import load_conv_bn, load_conv

def parse_cfg(cfgfile):
    blocks = []
    fp = open(cfgfile, 'r')
    block =  None
    line = fp.readline()
    while line != '':
        line = line.rstrip()
        if line == '' or line[0] == '#':
            line = fp.readline()
            continue        
        elif line[0] == '[':
            if block:
                blocks.append(block)
            block = dict()
            block['type'] = line.lstrip('[').rstrip(']')
            # set default value
            if block['type'] == 'convolutional':
                block['batch_normalize'] = 0
        else:
            key,value = line.split('=')
            key = key.strip()
            value = value.strip()
            block[key] = value
        line = fp.readline()

    if block:
        blocks.append(block)
    fp.close()
    return blocks

class Reorg(nn.Module):
    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        self.stride = stride

    def forward(self, x):
        stride = self.stride
        assert(x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        assert(H % stride == 0)
        assert(W % stride == 0)
        x = x.view(B, C, H/stride, stride, W/stride, stride).transpose(3, 4).contiguous()
        x = x.view(B, C, H*W/(stride*stride), stride*stride).transpose(2,3).contiguous()
        x = x.view(B, C*stride*stride, H/stride, W/stride)
        return x


class MaxPoolStride1(nn.Module):
    def __init__(self):
        super(MaxPoolStride1, self).__init__()

    def forward(self, x):
        x = F.max_pool2d(F.pad(x, (0,1,0,1), mode='replicate'), 2, stride=1)
        return x

class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        #print(self.blocks)
        self.model, self.loss = self.create_network(self.blocks)
        self.num_classes = self.loss.num_classes
        self.anchors = self.loss.anchors
        self.width = int(self.blocks[0]['width'])
        self.height = int(self.blocks[0]['height'])

    def forward(self, x):
        x = self.model(x)
        return x

    def create_network(self, blocks):
        model = nn.Sequential()
        loss = RegionLoss()
    
        prev_filters = 3
        conv_id = 0
        for block in blocks:
            if block['type'] == 'net':
                continue
            elif block['type'] == 'convolutional':
                conv_id = conv_id + 1
                batch_normalize = int(block['batch_normalize'])
                filters = int(block['filters'])
                kernel_size = int(block['size'])
                stride = int(block['stride'])
                is_pad = int(block['pad'])
                pad = (kernel_size-1)/2 if is_pad else 0
                activation = block['activation']
                if batch_normalize:
                    model.add_module('conv{0}'.format(conv_id), nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=False))
                    model.add_module('bn{0}'.format(conv_id), nn.BatchNorm2d(filters))
                else:
                    model.add_module('conv{0}'.format(conv_id), nn.Conv2d(prev_filters, filters, kernel_size, stride, pad))
                if activation == 'leaky':
                    model.add_module('leaky{0}'.format(conv_id), nn.LeakyReLU(0.1, inplace=True))
                prev_filters = filters
            elif block['type'] == 'maxpool':
                pool_size = int(block['size'])
                stride = int(block['stride'])
                if stride > 1:
                    model.add_module('pool{0}'.format(conv_id), nn.MaxPool2d(pool_size, stride))
                else:
                    model.add_module('pool{0}'.format(conv_id), MaxPoolStride1())
            elif block['type'] == 'region':
                anchors = block['anchors'].split(',')
                loss.anchors = [float(i) for i in anchors]
                loss.num_classes = int(block['classes'])
                loss.num_anchors = int(block['num'])
                assert(loss.num_anchors == len(loss.anchors)/2)
                loss.object_scale = float(block['object_scale'])
                loss.noobject_scale = float(block['noobject_scale'])
                loss.class_scale = float(block['class_scale'])
                loss.coord_scale = float(block['coord_scale'])
            else:
                print('unknown type %s' % (block['type']))
    
        return model, loss

    def load_weights(self, weightfile):
        ind = 0
        start = 4
        buf = np.fromfile(weightfile, dtype = np.float32)
        for block in self.blocks:
            if block['type'] == 'net':
                continue
            elif block['type'] == 'convolutional':
                batch_normalize = int(block['batch_normalize'])
                activation = block['activation']
                if batch_normalize:
                    start = load_conv_bn(buf, start, self.model[ind], self.model[ind+1])
                    ind = ind + 2
                else:
                    start = load_conv(buf, start, self.model[ind])
                    ind = ind+1
                if activation != 'linear':
                    ind = ind+1
            elif block['type'] == 'maxpool':
                ind = ind+1
            elif block['type'] == 'region':
                ind = ind+1
            else:
                print('unknown type %s' % (block['type']))
