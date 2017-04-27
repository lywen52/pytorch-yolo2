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
