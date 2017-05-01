### yolo2
#### Detection Using A Pre-Trained Model
```
wget http://pjreddie.com/media/files/yolo.weights
python detect.py cfg/yolo.cfg data/coco.names yolo.weights data/dog.jpg
```
You will see some output like this:
```
layer     filters    size              input                output
    0 conv     32  3 x 3 / 1   608 x 608 x   3   ->   608 x 608 x  32
    1 max          2 x 2 / 2   608 x 608 x  32   ->   304 x 304 x  32
    ......
   30 conv    425  1 x 1 / 1    19 x  19 x1024   ->    19 x  19 x 425
   31 detection
Loading weights from yolo.weights... Done!
data/dog.jpg: Predicted in 1.08085393906 seconds.
boat: 0.111234
boat: 0.095928
sheep: 0.247619
```

#### Real-Time Detection on a Webcam

#### Training YOLO on VOC

