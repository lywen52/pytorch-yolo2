import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from region_loss import RegionLoss
from utils import load_conv_bn, load_conv

def parse_cfg(cfgfile):
    blocks = []
    fp = open(cfgfile, 'r')
    block = dict()
    block['type'] = 'empty'
    line = fp.readline()
    while line != '':
        line = line.rstrip()
        if line == '' or line[0] == '#':
            line = fp.readline()
            continue        
        elif line[0] == '[':
            if block['type'] != 'empty':
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

    if block['type'] != 'empty':
        blocks.append(block)
    fp.close()
    return blocks

def create_darknet(blocks):
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

def load_darknet_weights(blocks, model, weightfile):
    ind = 0
    start = 4
    buf = np.fromfile(weightfile, dtype = np.float32)
    for block in blocks:
        if block['type'] == 'net':
            continue
        elif block['type'] == 'convolutional':
            batch_normalize = int(block['batch_normalize'])
            activation = block['activation']
            if batch_normalize:
                start = load_conv_bn(buf, start, model[ind], model[ind+1])
                ind = ind + 2
            else:
                start = load_conv(buf, start, model[ind])
                ind = ind+1
            if activation != 'linear':
                ind = ind+1
        elif block['type'] == 'maxpool':
            ind = ind+1
        elif block['type'] == 'region':
            ind = ind+1
        else:
            print('unknown type %s' % (block['type']))

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
        self.model, self.loss = create_darknet(self.blocks)
        print(self.model)
        self.num_classes = self.loss.num_classes
        self.anchors = self.loss.anchors

    def load_weights(self, weightfile):
        load_darknet_weights(self.blocks, self.model, weightfile)
#        buf = np.fromfile(weightfile, dtype = np.float32)
#        start = 4
#        
#        start = load_conv_bn(buf, start, self.model[0], self.model[1])
#        start = load_conv_bn(buf, start, self.model[4], self.model[5])
#        start = load_conv_bn(buf, start, self.model[8], self.model[9])
#        start = load_conv_bn(buf, start, self.model[12], self.model[13])
#        start = load_conv_bn(buf, start, self.model[16], self.model[17])
#        start = load_conv_bn(buf, start, self.model[20], self.model[21])
#        
#        start = load_conv_bn(buf, start, self.model[24], self.model[25])
#        start = load_conv_bn(buf, start, self.model[27], self.model[28])
#        start = load_conv(buf, start, self.model[30])


    def forward(self, x):
        x = self.model(x)
        return x
