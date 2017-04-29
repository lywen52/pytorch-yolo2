from __future__ import print_function
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torch.autograd import Variable
from tiny_yolo_face14 import TinyYoloFace14Net

import dataset
from utils import *
from region_loss import RegionLoss

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}

print('batch_size = %d' % (args.batch_size))
print('test_batch_size = %d' % (args.test_batch_size))

train_loader = torch.utils.data.DataLoader(
    dataset.listDataset('train.txt', shuffle=True,
                   transform=transforms.Compose([
                       transforms.Scale(160),
                       transforms.ToTensor(),
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    dataset.listDataset('test_small.txt', shuffle=False,
                   transform=transforms.Compose([
                       transforms.Scale(160),
                       transforms.ToTensor(),
                   ])),
    batch_size=args.batch_size, shuffle=False, **kwargs)

model = TinyYoloFace14Net()
region_loss = RegionLoss(model.num_classes, model.anchors)

#model.load_darknet_weights('face4.1nb_inc2_96.16.weights')

if args.cuda:
    model = torch.nn.DataParallel(model).cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

#cudnn.benchmark = True

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if (batch_idx+1) % 5 == 0:
            sys.stdout.write(".")
        if (batch_idx+1) % 100 == 0:
            print('')
        if (batch_idx+1) % 1000 == 0:
            test(epoch)
            model.train()

        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = region_loss(output, target)
        loss.backward()
        optimizer.step()
    adjust_learning_rate(optimizer, epoch)

def test(epoch):
    def truths_length(truths):
        for i in range(50):
            if truths[i][1] == 0:
                return i

    model.eval()
    num_classes = model.module.num_classes
    anchors = model.module.anchors

    conf_thresh = 0.25
    nms_thresh = 0.4
    iou_thresh = 0.5

    total = 0.0
    proposals = 0.0
    correct = 0.0

    lineId = 0
    for data, target in test_loader:
        if args.cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        output = model(data).data
        print('batch = %d' % (data.size(0)))
        for i in range(output.size(0)):
            lineId = lineId + 1

            boxes = get_region_boxes(output[i], conf_thresh, num_classes, anchors)
            boxes = nms(boxes, nms_thresh)
            truths = target[i].view(-1, 5)
            num_gts = truths_length(truths)

     
            total = total + num_gts
    
            for i in range(len(boxes)):
                if boxes[i][4] > conf_thresh:
                    proposals = proposals+1
    
            for i in range(num_gts):
                x1 = truths[i][1] - truths[i][3]/2.0
                y1 = truths[i][2] - truths[i][4]/2.0
                x2 = truths[i][1] + truths[i][3]/2.0
                y2 = truths[i][2] + truths[i][4]/2.0
                box_gt = [x1, y1, x2, y2, 1.0]
                best_iou = 0
                for j in range(len(boxes)):
                    iou = bbox_iou(box_gt, boxes[j])
                    best_iou = max(iou, best_iou)
                if best_iou > iou_thresh:
                    correct = correct+1
    
            precision = 1.0*correct/(proposals+0.000001)
            recall = 1.0*correct/(total+0.000001)
            fscore = 2.0*precision*recall/(precision+recall+0.000001)
            print("%d precision: %f, recal: %f, fscore: %f\n" % (lineId, precision, recall, fscore))

test(0)
for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
