# Hand Detection for IS

+ I've trained on about 587 images and tested on 66 images, the performance is not so good
+ I will use pretrained LISA model
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
    
## updated on 2017-08-29
+ The result of LISA-10740-ISIS-16200 is not good that the recall is too high
+ Till now there are several ways to modify:
    + Enlarge the dataset: currently there are only 500+ in training set and 66 in testing/CV set
        + Use data augmentation
        + Label more data
    + Use more robust basenet: e.g. ResNet
    + Seek other network: e.g. DenseNet, MaskRCNN
    
## updated on 2017-09-08
+ Download the model again and try to run on ResNet
+ Encounter error: [https://github.com/endernewton/tf-faster-rcnn/issues/107](https://github.com/endernewton/tf-faster-rcnn/issues/107)
```python
# Change lib/model/train_val.py Line278
blobs = self.data_layer.forward() 
if blobs['gt_boxes'][0][1] > blobs['gt_boxes'][0][3]:
    iter += 1    
    continue
```
