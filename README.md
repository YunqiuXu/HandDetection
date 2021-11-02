# HandDetection
Modify faster-rcnn for hand detection 

# Preprocessing
~/tf-faster-rcnn-endernewton/data/LISA_HD_Static/detectiondata$ python LISA_posGt_to_VOC_Annotations.py </br>
~/tf-faster-rcnn-endernewton/data/LISA_HD_Static/detectiondata$ python LISA_posGt_to_VOC_Main.py </br>

# Train
~/tf-faster-rcnn-endernewton$ ./experiments/scripts/train_faster_rcnn.sh 0 pascal_voc vgg16 </br>

# Test
Modifiy the iter times in test_faster_rcnn.sh </br>
~/tf-faster-rcnn-endernewton$ ./experiments/scripts/test_faster_rcnn.sh 0 pascal_voc vgg16 </br>

# How to do prediction on your own dataset

cd tf-faster-rcnn-endernewton/data/LISA_HD_Static/detectiondata/ImageSets/Main </br>
mv test.txt test_for_train.txt </br>
mv test5500.txt test.txt </br>

cd tf-faster-rcnn-endernewton/data/LISA_HD_Static/detectiondata </br>
mv JPEGImages JPEGImages_train </br>
mv JPEGImages_test JPEGImages </br>

Open tf-faster-rcnn-endernewton/experiments/scripts/test_faster_rcnn.sh </br>
Set line 21 "ITERS = the iters of the model you trained" Say if you trained a model with 10000 iters, set this line "ITERS = 10000" </br>

cd tf-faster-rcnn-endernewton </br>
./experiments/scripts/test_faster_rcnn.sh 0 pascal_voc vgg16 </br>

# How to stop the training

tmux attach </br>
ctrl+c

+ This project is collaboration with my collegue Yunqiu Xu (https://github.com/YunqiuXu).
 
