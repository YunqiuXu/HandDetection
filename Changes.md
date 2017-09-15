# Change Log

## Updated on 2017-08-24
+ Run on pre-trained LISA model
+ Prob 1(solved)
    + Different from load vgg16 directly, it seems that if I restore a model instead of loading vgg16 directly, the num_classes needs to be same with before, otherwise it will output InvalidArgumentError
    + So I change pascal_voc.py to add 2 useless classes
+ Prob 2(solved)
    + lib/roi_data_layer/layer.py line82 minibatch_db = [self._roidb[i] for i in db_inds] out of range
    + our training set 587*2 = 1174 roidb, testing set 66 roidb
    + lisa training set 5225, testing set 275
    + So we scale them as 
```python
roidb_size = len(self._roidb)
if roidb_size > 1000:
    minibatch_db = [self._roidb[int(i*roidb_size*1.0/5225)] for i in db_inds]
else:
    minibatch_db = [self._roidb[int(i*roidb_size*1.0/275)] for i in db_inds]
```

+ Use model pretrained on LISA to train on ISIS
    + Restored from LISA 10740 iters
    + Run on ISIS till 16200 iters
    
## Updated on 2017-08-29
+ The result of LISA-10740-ISIS-16200 is not good that the recall is too high
+ Till now there are several ways to modify:
    + Enlarge the dataset: currently there are only 500+ in training set and 66 in testing/CV set
        + Use data augmentation
        + Label more data
    + Use more robust basenet: e.g. ResNet
    + Seek other network: e.g. DenseNet, MaskRCNN
    
## Updated on 2017-09-08
+ Download the model again and try to run on ResNet
+ Prob 1(solved): [https://github.com/endernewton/tf-faster-rcnn/issues/107](https://github.com/endernewton/tf-faster-rcnn/issues/107)
    + change pascal_voc.py LINE167, remove all "-1"
```python
x1 = float(bbox.find('xmin').text)
y1 = float(bbox.find('ymin').text)
x2 = float(bbox.find('xmax').text)
y2 = float(bbox.find('ymax').text)
```
    + Maybe it's because GPU arch
        + Tesla K40c : sm_35
        + Quadro 620 : sm_50
+ Prob 2(solved): imdb.py "assert (boxes[:, 2] >= boxes[:, 0]).all()"
```python
for b in range(len(boxes)):
  if boxes[b][2]< boxes[b][0]:
    boxes[b][0] = 0
```        
+ Resnet-30000, result is decent --> crop
    + map 0.70
    + By standarize the images we can get better result!
    + however there maybe similar images among training and testing set

## updated on 2017-09-15
+ Reuse modified vgg16, run on resized images
+ Since tf-faster-rcnn changes a lot, I will not use previous vgg16.py any longer. Instead, I modify the new vgg16.py, and network.py to fit current program. For more details, see VGG16Modified
    + vgg16_modified.py
    + network_modified.py
+ Warning, for resnet101, you should rename network_original.py to network.py
    
    

