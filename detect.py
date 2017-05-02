import sys
import time
from PIL import Image, ImageDraw
from tiny_yolo import TinyYoloNet
from utils import do_detect, plot_boxes, load_class_names
from darknet import Darknet
from darknet2 import Darknet2

def detect(cfgfile, namesfile, weightfile, imgfile, version=2):
    if version == 1:
        m = Darknet(cfgfile) 
    elif version == 2:
        # support reorg, route
        m = Darknet2(cfgfile) 

    #print(m)
    m.print_network()
    m.float()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))
    
    use_cuda = 1
    if use_cuda:
        m.cuda()

    img = Image.open(imgfile).convert('RGB')
    sized = img.resize((m.width, m.height))
    
    start = time.time()
    boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
    finish = time.time()
    print('%s: Predicted in %s seconds.' % (imgfile, (finish-start)))

    class_names = load_class_names(namesfile)
    plot_boxes(img, boxes, 'predictions.jpg', class_names)

############################################
if __name__ == '__main__':
    # with/without route layer
    if len(sys.argv) == 5:
        cfgfile = sys.argv[1]
        namesfile = sys.argv[2]
        weightfile = sys.argv[3]
        imgfile = sys.argv[4]
        detect(cfgfile, namesfile, weightfile, imgfile, version=2)
    else:
        detect('cfg/tiny-yolo-voc.cfg', 'data/voc.names', 'tiny-yolo-voc.weights', 'data/person.jpg', version=1)
