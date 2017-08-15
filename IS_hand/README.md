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
