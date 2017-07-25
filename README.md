# Modified Faster-RCNN for Hand Detection
Author: Yunqiu Xu, Shaoshen Wang

## Changes
### data/LISA_HD_Static
        + Download and unzip data from http://cvrr.ucsd.edu/vivachallenge/index.php/hands/hand-detection/
        + move 'LISA_annotation_to_VOC.py' and 'LISA_posGt_to_VOC_main.py' to /LISA_HD_Static/detectiondata
        + LISA_annotation_to_VOC.py  -- change dataset to VOC2007 form
        + LISA_posGt_to_VOC_main.py  -- split the dataset into train/cv/test, files are in ImageSets/Main
        + Thus we generate 4 txt file train.txt val.txt trainval.txt test.txt under Main
        
```python
# VOC2007 download from http://people.csail.mit.edu/tomasz/VOCdevkit/
- VOC2007
    - Annotations/
        - 000001.xml
        - 000002.xml
        - ...
    - ImageSets/
        - Layout/
            - test.txt # the label of testing data, e.g. '000001'
            - train.txt # the label of  training data, e.g. '000002'
            - trainval.txt # cv labels
            - val # cv labels
        - Main/
            # for each class we have test/train/trainval/val
            - aeroplane_test.txt # all labels, if it's this class --> 1, else --> -1
            - aeroplane_train.txt
            - aeroplane_trainval.txt
            - aeroplane_val.txt
            - ...
        - Segmentation/
            - test.txt 
            - train.txt 
            - trainval.txt 
            - val.txt
    - JPEGImages/
        - 000001.jpg
        - ...
    - SegmentationClass/
        - 000032.png # same to labels of Segmentation
    - SegmentationObject/
        - 000032.png 
```       
        
        
### data/cache
        + if you change model or training set, you need delete this folder
        + previous files may be saved in this folder, we should not load them in new loops
### lib/model/config.py
        + __C.TRAIN.USE_FLIPPED = False -- do not use data augmentation
        + We can change other paramaters as well(e.g. learning rate)
### lib/datasets/pascal_voc.py
        + line 43: change the classes        
        + line 47: '.jpg' --> '.png'
        + line 167-170: remove '-1' -- original start position is (1,1), now (0,0)
        + line 180: remove '.lower()' -- otherwise some names will be same
### lib/nets/vgg16.py
        + Most changes are based on vgg16 model
        + line 319: some weights appeared before need to be reloaded
### lib/nets/network.py
        + line 30: self._feat_stride = [4, ]
        + line 33: self._feat_compress = [1. / 4., ]
        + Reason: in original version the shape is 14*14*n, while it's 56*56*n in modified vgg16.py, so the ratio is 224/56 = 4
        
### lib/datasets/voc_eval.py
        + define a new function 'transform(line,hand,person)'
        + some codes are omitted, because we do not need classification result but the submission file
        + if we need to get result, we should use unchanged voc_eval.py

## Training Process
```shell
~/tf-faster-rcnn-endernewton$ ./experiments/scripts/train_faster_rcnn.sh 0 pascal_voc vgg16
```
+ For every 20 iters a model will be built, only the latest 3 models will be saved

## Testing Process
+ Modifiy the iter times in test_faster_rcnn.sh
        + Check the models in output/vgg16/voc_2007_trainval/default
        + If the model is vgg16_faster_rcnn_iter_10740.pkl, we need to set iter times as 10740
```shell
~/tf-faster-rcnn-endernewton$ ./experiments/scripts/test_faster_rcnn.sh 0 pascal_voc vgg16 
```
+ The result consists of 4 txt files in result folder, then we can run combine_result.py to combine them --> detection_result.txt


