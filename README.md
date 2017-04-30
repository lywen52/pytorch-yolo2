#### Detection Using A Pre-Trained Model
```
wget http://pjreddie.com/media/files/tiny-yolo-voc.weights
python detect.py cfg/yolo.cfg data/coco.names yolo.weights data/dog.jpg
```
You will see some output like this:
```
layer     filters    size              input                output
    0 conv     32  3 x 3 / 1   416 x 416 x   3   ->   416 x 416 x  32
    1 max          2 x 2 / 2   416 x 416 x  32   ->   208 x 208 x  32
    .......
   29 conv    425  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 425
   30 detection
Loading weights from yolo.weights...Done!
data/dog.jpg: Predicted in 0.016287 seconds.
car: 54%
bicycle: 51%
dog: 56%
```



#### yolo2
This repository can evaluate darknent trained model on pytorch which loads darkent trained weight file directly. See test_tiny_yolo.py as an example.

#### Todo
- [x] load darknet weights to pytorch
- [x] evaluate tiny-yolo on single image
- [x] evaluate tiny-yolo on image list
- [ ] evaluate tiny-yolo on ImageList Dataset
- [ ] train model on ImageList Dataset

#### Note
1. be sure to add m.eval()
2. running_var is processed differently
   - darkent: sqrt(running_var) + 0.00001
   - pytorch: sqrt(running_var + 0.00001)

####
batch
10000 precision: 0.957715, recal: 0.960783, fscore: 0.959246

list
10000 precision: 0.955355, recal: 0.961475, fscore: 0.958405
