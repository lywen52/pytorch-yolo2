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

def create_darknet_simple(blocks):
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

def create_darknet(blocks):
    models = nn.ModuleList()

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
            model = nn.Sequential()
            if batch_normalize:
                model.add_module('conv{0}'.format(conv_id), nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=False))
                model.add_module('bn{0}'.format(conv_id), nn.BatchNorm2d(filters))
            else:
                model.add_module('conv{0}'.format(conv_id), nn.Conv2d(prev_filters, filters, kernel_size, stride, pad))
            if activation == 'leaky':
                model.add_module('leaky{0}'.format(conv_id), nn.LeakyReLU(0.1, inplace=True))
            prev_filters = filters
            models.append(model)
        elif block['type'] == 'maxpool':
            pool_size = int(block['size'])
            stride = int(block['stride'])
            if stride > 1:
                model = nn.MaxPool2d(pool_size, stride)
            else:
                model = MaxPoolStride1()
            models.append(model)
        elif block['type'] == 'region':
            loss = RegionLoss()
            anchors = block['anchors'].split(',')
            loss.anchors = [float(i) for i in anchors]
            loss.num_classes = int(block['classes'])
            loss.num_anchors = int(block['num'])
            assert(loss.num_anchors == len(loss.anchors)/2)
            loss.object_scale = float(block['object_scale'])
            loss.noobject_scale = float(block['noobject_scale'])
            loss.class_scale = float(block['class_scale'])
            loss.coord_scale = float(block['coord_scale'])
            models.append(model)
        else:
            print('unknown type %s' % (block['type']))

    return models


def load_darknet_weights_simple(blocks, model, weightfile):
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

def load_darknet_weights(blocks, models, weightfile):
    ind = 0
    start = 4
    buf = np.fromfile(weightfile, dtype = np.float32)
    ind = -2
    for block in blocks:
        ind = ind + 1
        if block['type'] == 'net':
            continue
        elif block['type'] == 'convolutional':
            model = models[ind]
            batch_normalize = int(block['batch_normalize'])
            if batch_normalize:
                start = load_conv_bn(buf, start, model[0], model[1])
            else:
                start = load_conv(buf, start, model[0])
        elif block['type'] == 'maxpool':
            ind = ind
        elif block['type'] == 'region':
            ind = ind
        else:
            print('unknown type %s' % (block['type']))

class MaxPoolStride1(nn.Module):
    def __init__(self):
        super(MaxPoolStride1, self).__init__()

    def forward(self, x):
        x = F.max_pool2d(F.pad(x, (0,1,0,1), mode='replicate'), 2, stride=1)
        return x

class DarknetSimple(nn.Module):
    def __init__(self, cfgfile):
        super(DarknetSimple, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        #print(self.blocks)
        self.model, self.loss = create_darknet_simple(self.blocks)
        print(self.model)
        self.num_classes = self.loss.num_classes
        self.anchors = self.loss.anchors

    def load_weights(self, weightfile):
        load_darknet_weights_simple(self.blocks, self.model, weightfile)

    def forward(self, x):
        x = self.model(x)
        return x

class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet,self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.models = create_darknet(self.blocks) # merge conv, bn,leaky
        self.loss = 0
        for block in self.blocks:
            if block['type'] == 'region':
                anchors = block['anchors'].split(',')
                self.num_classes = int(block['classes'])
                self.anchors = [float(i) for i in anchors]

    def load_weights(self, weightfile):
        load_darknet_weights(self.blocks, self.models, weightfile)
    def forward(self, x):
        #for model in self.models:
        for i in range(len(self.models)-1):
            x = self.models[i](x)
        return x

    def forward2(self, x):
        ind = -2
        self.loss = 0
        outputs = dict()
        for block in self.blocks:
            ind = ind + 1
            if block['type'] == 'net':
                continue
            elif block['type'] == 'convolutional' or block['type'] == 'maxpool':
                x = self.models(x)
                outputs[ind] = x
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                layers = [int(i) if int(i) > 0 else int(i)+ind for i in layers]
                if len(layers) == 1:
                    x = outputs[layers[0]]
                    outputs[ind] = x
                elif len(layers) == 2:
                    x1 = outputs[layers[0]]
                    x2 = outputs[layers[1]]
                    x = torch.cat(x1,x2,1)
                    outputs[ind] = x
            elif block['type'] == 'region':
                x = self.models(x)
                outputs[ind] = x
                if self.loss != 0:
                    self.loss = self.loss + x
                else:
                    self.loss = self.loss + x
        return x
