#### Detection Using A Pre-Trained Model
```
wget http://pjreddie.com/media/files/yolo.weights
python detect.py cfg/yolo.cfg data/coco.names yolo.weights data/dog.jpg
```
You will see some output like this:
```
Darknet2 (
  (models): ModuleList (
    (0): Sequential (
      (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True)
      (leaky1): LeakyReLU (0.1, inplace)
    )
    (1): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
    ......
    (28): ReLU ()
    (29): Sequential (
      (conv22): Conv2d(1280, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn22): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True)
      (leaky22): LeakyReLU (0.1, inplace)
    )
    (30): Sequential (
      (conv23): Conv2d(1024, 425, kernel_size=(1, 1), stride=(1, 1))
    )
    (31): RegionLoss (
    )
  )
)
Loading weights from yolo.weights... Done!
data/dog.jpg: Predicted in 1.054e-05 seconds.
boat: 0.111234
boat: 0.095928
sheep: 0.247619
```



#### yolo2
This repository can evaluate darknent trained model on pytorch which loads darkent trained weight file directly. See test_tiny_yolo.py as an example.

