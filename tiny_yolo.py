import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from utils import *

class TinyYoloNet(nn.Module):
    def __init__(self):
        super(TinyYoloNet, self).__init__()
        self.num_classes = 20
        self.anchors = [1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52]

        self.cnn1 = nn.Sequential(OrderedDict([
            # conv1
            ('conv1', nn.Conv2d( 3, 16, 3, 1, 1, bias=False)),
            ('bn1', nn.BatchNorm2d(16)),
            ('leaky1', nn.LeakyReLU(0.1, inplace=True)),
            ('pool1', nn.MaxPool2d(2, 2)),

            # conv2
            ('conv2', nn.Conv2d(16, 32, 3, 1, 1, bias=False)),
            ('bn2', nn.BatchNorm2d(32)),
            ('leaky2', nn.LeakyReLU(0.1, inplace=True)),
            ('pool2', nn.MaxPool2d(2, 2)),

            # conv3
            ('conv3', nn.Conv2d(32, 64, 3, 1, 1, bias=False)),
            ('bn3', nn.BatchNorm2d(64)),
            ('leaky3', nn.LeakyReLU(0.1, inplace=True)),
            ('pool3', nn.MaxPool2d(2, 2)),

            # conv4
            ('conv4', nn.Conv2d(64, 128, 3, 1, 1, bias=False)),
            ('bn4', nn.BatchNorm2d(128)),
            ('leaky4', nn.LeakyReLU(0.1, inplace=True)),
            ('pool4', nn.MaxPool2d(2, 2)),

            # conv5
            ('conv5', nn.Conv2d(128, 256, 3, 1, 1, bias=False)),
            ('bn5', nn.BatchNorm2d(256)),
            ('leaky5', nn.LeakyReLU(0.1, inplace=True)),
            ('pool5', nn.MaxPool2d(2, 2)),

            # conv6
            ('conv6', nn.Conv2d(256, 512, 3, 1, 1, bias=False)),
            ('bn6', nn.BatchNorm2d(512)),
            ('leaky6', nn.LeakyReLU(0.1, inplace=True)),
        ]))

        num_anchors = len(self.anchors)/2
        num_output = (5+self.num_classes)*num_anchors

        self.cnn2 = nn.Sequential(OrderedDict([
            # conv7
            ('conv7', nn.Conv2d(512, 1024, 3, 1, 1, bias=False)),
            ('bn7', nn.BatchNorm2d(1024)),
            ('leaky7', nn.LeakyReLU(0.1, inplace=True)),

            # conv8
            ('conv8', nn.Conv2d(1024, 1024, 3, 1, 1, bias=False)),
            ('bn8', nn.BatchNorm2d(1024)),
            ('leaky8', nn.LeakyReLU(0.1, inplace=True)),

            # output
            ('output', nn.Conv2d(1024, num_output, 1, 1, 0)),
        ]))

    def forward(self, x):
        x = F.max_pool2d(F.pad(self.cnn1(x), (0,1,0,1), mode='replicate'), 2, stride=1)
        x = self.cnn2(x)
        return x
        #return F.log_softmax(x)

    def load_darknet_weights(self, path):
        #buf = np.fromfile('tiny-yolo-voc.weights', dtype = np.float32)
        buf = np.fromfile(path, dtype = np.float32)
        start = 4
        
        start = load_conv_bn(buf, start, self.cnn1[0], self.cnn1[1])
        start = load_conv_bn(buf, start, self.cnn1[4], self.cnn1[5])
        start = load_conv_bn(buf, start, self.cnn1[8], self.cnn1[9])
        start = load_conv_bn(buf, start, self.cnn1[12], self.cnn1[13])
        start = load_conv_bn(buf, start, self.cnn1[16], self.cnn1[17])
        start = load_conv_bn(buf, start, self.cnn1[20], self.cnn1[21])
        
        start = load_conv_bn(buf, start, self.cnn2[0], self.cnn2[1])
        start = load_conv_bn(buf, start, self.cnn2[3], self.cnn2[4])
        start = load_conv(buf, start, self.cnn2[6])

