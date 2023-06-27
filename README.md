# Master thesis code Jesse Kassai
This README is copied from the original source (https://github.com/yjh0410/new-YOLOv1_PyTorch/blob/master/models/yolo.py), only this block is added to give additional explanations for the thesis research. 
The data can be downloaded via the instructions in this file, but for open set recognition, a split needs to be made into the datasets. 

- With the file "open_test_set.py" you can select the classes you want to keep in the open set evaluation and the classes you need to keep for training the model on the open-set setting. For this you need to uncomment the commented part of this code and comment the uncommented part of this code.
- After the first step you need to use "annot_remove.py" to filter out the annotations for the classes that are not in the open-set evaluation and open-set training anymore.
- After making the splits you can use the file "convert.py" to match the train.txt, trainval.txt and test.txt files with the annotations that are now changed because of the open-set conversion. For this you need to change the file path of line 35 to the aforementioned files. 

To switch between the various modes in open set training and evaluation & closed-set training and evaluation, there are a few settings that needs to be adjusted. 
- In the file "yolo.py", you need to set the parameter "open" to True. Also in these parameters you need to set the mode to "oh" (original model), "neck" (hyperbolic neck model) or "hyp" (hyperbolic head model). For open-set you also need to adjust this in "train_open.py" at line 159.
- In the file "voc0712.py", you need to change the "VOC_ROOT" path to ```path_to_dir + "/VOCdevkit"``` for training and  to ```path_to_dir + "/VOCdevkit/VOC2007OPENPART2.2"``` for evaluation. Also in this file, you need to set the right VOC_CLASSES for both training and evaluation by commenting out the right classes.
- In the file "vocapi_evaluator.py", you an set the confidence threshold at line 177. Also to switch between closed- and open-set setting, you need to switch between lines 37-41 and 44-48 respectively.
- In the file "eval.py", you need to set the right amount of classes corresponding to the number of classes used for the task you are executing (for example: 3 if you are evaluating on open-set setting with 3 unseen classses).

To train on closed set (Pascal VOC: voc, MS COCO: coco),  you run the following command: ```python train.py --cuda -d voc```
To train on open set (Pascal VOC: voc, MS COCO: coco),  you run the following command: ```python train_open.py --cuda -d voc```

# new-YOLOv1_PyTorch
In this project, you can enjoy: 
- a new version of yolov1


# Network
This is a a new version of YOLOv1 built by PyTorch:
- Backbone: resnet18
- Head: SPP, SAM

# Train
- Batchsize: 32
- Base lr: 1e-3
- Max epoch: 160
- LRstep: 60, 90
- optimizer: SGD

Before I tell you how to use this project, I must say one important thing about difference between origin yolo-v2 and mine:

- For data augmentation, I copy the augmentation codes from the https://github.com/amdegroot/ssd.pytorch which is a superb project reproducing the SSD. If anyone is interested in SSD, just clone it to learn !(Don't forget to star it !)

So I don't write data augmentation by myself. I'm a little lazy~~

My loss function and groundtruth creator both in the ```tools.py```, and you can try to change any parameters to improve the model.

## Experiment
Environment:

- Python3.6, opencv-python, PyTorch1.1.0, CUDA10.0,cudnn7.5
- For training: Intel i9-9940k, TITAN-RTX-24g
- For inference: Intel i5-6300H, GTX-1060-3g

VOC:
<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8> </th>     <td bgcolor=white> size </td><td bgcolor=white> mAP </td><td bgcolor=white> FPS </td></tr>
<tr><th align="left" bgcolor=#f8f8f8> VOC07 test</th><td bgcolor=white> 320 </td><td bgcolor=white> 64.4 </td><td bgcolor=white> - </td></tr>
<tr><th align="left" bgcolor=#f8f8f8> VOC07 test</th><td bgcolor=white> 416 </td><td bgcolor=white> 68.5 </td><td bgcolor=white> - </td></tr>
<tr><th align="left" bgcolor=#f8f8f8> VOC07 test</th><td bgcolor=white> 608 </td><td bgcolor=white> 71.5 </td><td bgcolor=white> - </td></tr>
</table></tbody>

COCO:
<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8> </th>     <td bgcolor=white> size </td><td bgcolor=white> AP </td><td bgcolor=white> AP50 </tr>
<tr><th align="left" bgcolor=#f8f8f8> COCO val</th><td bgcolor=white> 320 </td><td bgcolor=white> 14.50 </td><td bgcolor=white> 30.15 </td></tr>
<tr><th align="left" bgcolor=#f8f8f8> COCO val</th><td bgcolor=white> 416 </td><td bgcolor=white> 17.34 </td><td bgcolor=white> 35.28 </td></tr>
<tr><th align="left" bgcolor=#f8f8f8> COCO val</th><td bgcolor=white> 608 </td><td bgcolor=white> 19.90 </td><td bgcolor=white> 39.27 </td></tr>
</table></tbody>

## Installation
- Pytorch-gpu 1.1.0/1.2.0/1.3.0
- Tensorboard 1.14.
- opencv-python, python3.6/3.7

## Dataset
As for now, I only train and test on PASCAL VOC2007 and 2012. 

### VOC Dataset
I copy the download files from the following excellent project:
https://github.com/amdegroot/ssd.pytorch

I have uploaded the VOC2007 and VOC2012 to BaiDuYunDisk, so for researchers in China, you can download them from BaiDuYunDisk:

Link：https://pan.baidu.com/s/1tYPGCYGyC0wjpC97H-zzMQ 

Password：4la9

You will get a ```VOCdevkit.zip```, then what you need to do is just to unzip it and put it into ```data/```. After that, the whole path to VOC dataset is:

- ```data/VOCdevkit/VOC2007```
- ```data/VOCdevkit/VOC2012```.

#### Download VOC2007 trainval & test

```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2007.sh # <directory>
```

#### Download VOC2012 trainval
```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2012.sh # <directory>
```

### MSCOCO Dataset
I copy the download files from the following excellent project:
https://github.com/DeNA/PyTorch_YOLOv3

#### Download MSCOCO 2017 dataset
Just run ```sh data/scripts/COCO2017.sh```. You will get COCO train2017, val2017, test2017:

- ```data/COCO/annotations/```
- ```data/COCO/train2017/```
- ```data/COCO/val2017/```
- ```data/COCO/test2017/```


# Train
### VOC
```Shell
python train.py -d voc --cuda -v [select a model] -ms
```

You can run ```python train.py -h``` to check all optional argument.

### COCO
```Shell
python train.py -d coco --cuda -v [select a model] -ms
```


## Test
### VOC
```Shell
python test.py -d voc --cuda -v [select a model] --trained_model [ Please input the path to model dir. ]
```

### COCO
```Shell
python test.py -d coco-val --cuda -v [select a model] --trained_model [ Please input the path to model dir. ]
```


## Evaluation
### VOC
```Shell
python eval.py -d voc --cuda -v [select a model] --train_model [ Please input the path to model dir. ]
```

### COCO
To run on COCO_val:
```Shell
python eval.py -d coco-val --cuda -v [select a model] --train_model [ Please input the path to model dir. ]
```

To run on COCO_test-dev(You must be sure that you have downloaded test2017):
```Shell
python eval.py -d coco-test --cuda -v [select a model] --train_model [ Please input the path to model dir. ]
```
You will get a .json file which can be evaluated on COCO test server.
