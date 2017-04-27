import os
import torch
import math
import numpy as np
from PIL import Image, ImageDraw
from torch.autograd import Variable
from tiny_yolo import TinyYoloNet
from tiny_yolo_face14 import TinyYoloFace14Net
from utils import *

use_cuda = 1

def do_detect(model, img, conf_thresh, nms_thresh):
    model.eval()
    width = img.width
    height = img.height
    img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
    img = img.view(height, width, 3).transpose(0,1).transpose(0,2).contiguous()
    img = img.view(1, 3, height, width)
    img = img.float().div(255.0)
    if use_cuda:
        img = img.cuda()
    img = Variable(img)

    output = model(img)
    output = output.data
    boxes = get_region_boxes(output, conf_thresh, model.num_classes, model.anchors)
    boxes = nms(boxes, nms_thresh)
    return boxes
   
def demo(tiny_yolo_weight, img_path):
    m = TinyYoloNet() 
    m.float()
    m.eval()
    m.load_darknet_weights(tiny_yolo_weight)
    
    if use_cuda:
        m.cuda()

    img = Image.open(img_path).convert('RGB')
    sized = img.resize((416,416))
    boxes = do_detect(m, sized, 0.5, 0.4)
    plot_boxes(img, boxes, 'predict.jpg')    

def eval_list(tiny_yolo_weight, img_list, eval_wid, eval_hei):
    m = TinyYoloFace14Net()
    m.float()
    m.eval()
    m.load_darknet_weights(tiny_yolo_weight)

    if use_cuda:
        m.cuda()

    conf_thresh = 0.25
    nms_thresh = 0.4
    iou_thresh = 0.5

    with open(img_list) as fp:
        lines = fp.readlines()

    total = 0.0
    proposals = 0.0
    correct = 0.0
    lineId = 0
    for line in lines:
        lineId = lineId + 1
        img_path = line.rstrip()
        lab_path = img_path.replace('images', 'labels').replace('.jpg', '.txt')
        truths = read_truths(lab_path)
        #print(truths)

        img = Image.open(img_path).convert('RGB').resize((eval_wid, eval_hei))
        boxes = do_detect(m, img, conf_thresh, nms_thresh)
        savename = "tmp/%06d.jpg" % (lineId)
        print("save %s" % savename)
        plot_boxes(img, boxes, savename)
        
        total = total + truths.shape[0]

        for i in range(len(boxes)):
            if boxes[i][4] > conf_thresh:
                proposals = proposals+1

        for i in range(truths.shape[0]):
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

        precision = 1.0*correct/proposals
        recall = 1.0*correct/total
        fscore = 2.0*precision*recall/(precision+recall)
        print("%d precision: %f, recal: %f, fscore: %f\n" % (lineId, precision, recall, fscore))

############################################
if __name__ == '__main__':
    #demo('tiny-yolo-voc.weights', 'person.jpg')
    eval_list('face4.1nb_inc2_96.16.weights', 'test.txt', 160, 160)
