import os
import math
import torch
import numpy as np
from PIL import Image, ImageDraw

def sigmoid(x):
    return 1.0/(math.exp(-x)+1.)

def softmax(x):
    max_x = torch.max(x)
    x = x - max_x
    x = torch.exp(x)
    return x/x.sum()


def bbox_iou(box1, box2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        Mx = max(box1[2], box2[2])
        my = min(box1[1], box2[1])
        My = max(box1[3], box2[3])
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else:
        mx = min(box1[0]-box1[2]/2.0, box2[0]-box2[2]/2.0)
        Mx = max(box1[0]+box1[2]/2.0, box2[0]+box2[2]/2.0)
        my = min(box1[1]-box1[3]/2.0, box2[1]-box2[3]/2.0)
        My = max(box1[1]+box1[3]/2.0, box2[1]+box2[3]/2.0)
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    carea = 0
    if cw <= 0 or ch <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    uarea = area1 + area2 - carea
    return carea/uarea

def nms(boxes, nms_thresh):
    if len(boxes) == 0:
        return boxes

    det_confs = torch.zeros(len(boxes))
    for i in range(len(boxes)):
        det_confs[i] = 1-boxes[i][4]                

    _,sortIds = torch.sort(det_confs)
    out_boxes = []
    for i in range(len(boxes)):
        box_i = boxes[sortIds[i]]
        if box_i[4] > 0:
            out_boxes.append(box_i)
            for j in range(i+1, len(boxes)):
                box_j = boxes[sortIds[j]]
                if bbox_iou(box_i, box_j) > nms_thresh:
                    box_j[4] = 0
    return out_boxes

def get_region_boxes(output, conf_thresh, num_classes, anchors):
    num_anchors = len(anchors)/2
    if output.dim() == 3:
        output = output.unsqueeze(0)
    assert(output.size(0) == 1)
    assert(output.size(1) == (5+num_classes)*num_anchors)
    h = output.size(2)
    w = output.size(3)
    boxes = []
    for cy in range(h):
        for cx in range(w):
            for i in range(num_anchors):
                start = (5+num_classes)*i
                bcx = sigmoid(output[0][start][cy][cx]) + cx
                bcy = sigmoid(output[0][start+1][cy][cx]) + cy
                bw = anchors[2*i] * math.exp(output[0][start+2][cy][cx])
                bh = anchors[2*i+1] * math.exp(output[0][start+3][cy][cx])
                det_conf = sigmoid(output[0][start+4][cy][cx]) 
                cls_confs = softmax(output[0][start+5:start+5+num_classes][cy][cx])
                cls_conf, cls_id = torch.max(cls_confs, 0)
                cls_conf = cls_conf[0]
                cls_id = cls_id[0]
                x1 = bcx - bw/2
                y1 = bcy - bh/2
                x2 = bcx + bw/2
                y2 = bcy + bh/2
                x1 = max(x1, 0.0)
                y1 = max(y1, 0.0)
                x2 = min(x2, w)
                y2 = min(y2, h)
                if det_conf > conf_thresh:
                    box = [x1/w, y1/h, x2/w, y2/h, det_conf, cls_conf, cls_id]
                    boxes.append(box)
    return boxes

def plot_boxes(img, boxes, savename, class_names = None):
    width = img.width
    height = img.height
    draw = ImageDraw.Draw(img)
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = box[0] * width
        y1 = box[1] * height
        x2 = box[2] * width
        y2 = box[3] * height
        draw.rectangle([x1, y1, x2, y2])
        if len(box) >= 7:
            cls_conf = box[5]
            cls_id = box[6]
            print('%s: %f' % (class_names[cls_id], cls_conf))
    img.save(savename)

def load_conv(buf, start, conv_model):
    num_w = conv_model.weight.numel()
    num_b = conv_model.bias.numel()
    conv_model.bias.data.copy_(torch.from_numpy(buf[start:start+num_b])); start = start + num_b
    conv_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_w])); start = start + num_w
    return start

def load_conv_bn(buf, start, conv_model, bn_model):
    num_w = conv_model.weight.numel()
    num_b = bn_model.bias.numel()
    bn_model.bias.data.copy_(torch.from_numpy(buf[start:start+num_b])); start = start + num_b
    bn_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_b])); start = start + num_b
    bn_model.running_mean.copy_(torch.from_numpy(buf[start:start+num_b])); start = start + num_b
    running_var = torch.from_numpy(buf[start:start+num_b]); start = start + num_b
    bn_model.running_var.copy_(running_var)
    conv_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_w])); start = start + num_w 
    return start

def load_fc(buf, start, fc_model):
    num_w = fc_model.weight.numel()
    num_b = fc_model.bias.numel()
    fc_model.bias.data.copy_(torch.from_numpy(buf[start:start+num_b])); start = start + num_b
    fc_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_w])); start = start + num_w 
    return start

def read_truths(lab_path):
    if os.path.getsize(lab_path):
        truths = np.loadtxt(lab_path)
        truths = truths.reshape(truths.size/5, 5)
        return truths
    else:
        return np.array([])

def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names

def do_detect(model, img, conf_thresh, nms_thresh, use_cuda=1):
    model.eval()
    if isinstance(img, Image.Image):
        width = img.width
        height = img.height
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
        img = img.view(height, width, 3).transpose(0,1).transpose(0,2).contiguous()
        img = img.view(1, 3, height, width)
        img = img.float().div(255.0)
    if use_cuda:
        img = img.cuda()
    img = torch.autograd.Variable(img)

    output = model(img)
    output = output.data
    boxes = get_region_boxes(output, conf_thresh, model.num_classes, model.anchors)
    boxes = nms(boxes, nms_thresh)
    return boxes

