from PIL import Image, ImageDraw
from tiny_yolo import TinyYoloNet
from utils import *
from darknet import Darknet
from darknet2 import Darknet2

use_cuda = 1
   
def demo1(tiny_yolo_weight, img_path):
    m = TinyYoloNet() 
    m.float()
    m.eval()
    m.load_darknet_weights(tiny_yolo_weight)
    
    if use_cuda:
        m.cuda()

    img = Image.open(img_path).convert('RGB')
    sized = img.resize((416,416))
    boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
    plot_boxes(img, boxes, 'predict1.jpg')    

def demo2(cfgfile, weightfile, img_path):
    m = Darknet(cfgfile) 
    m.float()
    m.load_weights(weightfile)
    m.eval()
    
    if use_cuda:
        m.cuda()

    img = Image.open(img_path).convert('RGB')
    sized = img.resize((416,416))
    boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
    plot_boxes(img, boxes, 'predict2.jpg')    

def demo3(cfgfile, weightfile, img_path):
    m = Darknet2(cfgfile) 
    m.float()
    m.load_weights(weightfile)
    m.eval()
    
    if use_cuda:
        m.cuda()

    img = Image.open(img_path).convert('RGB')
    sized = img.resize((416,416))
    boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
    plot_boxes(img, boxes, 'predict3.jpg')    



############################################
if __name__ == '__main__':
    demo1('tiny-yolo-voc.weights', 'data/person.jpg')
    demo2('cfg/tiny-yolo-voc.cfg', 'tiny-yolo-voc.weights', 'data/person.jpg')
    demo3('cfg/yolo-voc.cfg', 'yolo-voc.weights', 'data/person.jpg')
