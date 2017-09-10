# Preprocessing
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

+ Step 7: Data augmentation
    + Crop 224 \* 224 from each image --> 1024 images
    + Reset the labels, threshold = 0.5: if less than half of this label is not in cropped image, the label will be removed from this
