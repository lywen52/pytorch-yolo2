from darknet import Darknet

def partial(cfgfile, weightfile, outfile, max_layer):
    m = Darknet(cfgfile)
    m.float()
    m.load_weights(weightfile)
    m.save_weights(outfile, max_layer)
    print('save %s' % (outfile))

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 5:
        cfgfile = sys.argv[1]
        weightfile = sys.argv[2]
        outfile = sys.argv[3]
        max_layer = int(sys.argv[4])
        partial(cfgfile, weightfile, outfile, max_layer)
    else:
        partial('cfg/tiny-yolo-voc.cfg', 'tiny-yolo-voc.weights', 'tiny-yolo-voc.conv.15', 15)
