# Preprocessing

+ VOC2007 format:     
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

+ Step 1: Download the data from google image, the parameters can be set in config.json
```shell
python google_image_crawler.py
```

+ Step 2: Rename the *.jpg files to 000001.jpg
```shell
./rename.sh
```

+ Step 3: Resize the images to 224 \* 480 (same with ResNet)
```shell
python resize.py
```

+ Step 4: Label the images using LabelIMG, put xml files in old_annotations

+ Step 5: Create folder Annotations, change xml files to VOC2007 format and put them into this folder
```shell
python labelIMG_to_VOC2007.py
```

+ Step 6: Create folder ImageSets/Main, then build training / CV / testing set
```shell
python build_main.py
```
