# Modified Faster-RCNN for Hand Detection

+ Setup via [https://github.com/endernewton/tf-faster-rcnn](https://github.com/endernewton/tf-faster-rcnn) make sure you can run the demo

+ Modify the code via [Robust Hand Detection in Vehicles](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7899695) for hand detection. See Changes.md to modify the code

## How to run     
+ data/cache
    + if you change model or training set, you need delete this folder
    + previous files may be saved in this folder, we should not load them in new loops

+ Training Process
```shell
~/tf-faster-rcnn-endernewton$ ./experiments/scripts/train_faster_rcnn.sh 0 pascal_voc vgg16(or res101)
```

+ Testing Process
    + Change the test.txt if needed
    + Modifiy the iter times in test_faster_rcnn.sh
        + Check the models in output/vgg16/voc_2007_trainval/default
        + If the model is vgg16_faster_rcnn_iter_10740.pkl, we need to set iter times as 10740
```shell
~/tf-faster-rcnn-endernewton$ ./experiments/scripts/test_faster_rcnn.sh 0 pascal_voc vgg16(or res101)
```
+ The result is in result folder


