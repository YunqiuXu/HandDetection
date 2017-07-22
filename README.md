# HandDetection
Modify faster-rcnn for hand detection 

# How to do prediction on your own dataset

cd tf-faster-rcnn-endernewton/data/LISA_HD_Static/detectiondata/ImageSets/Main
mv test.txt test_for_train.txt
mv test5500.txt test.txt

cd tf-faster-rcnn-endernewton/data/LISA_HD_Static/detectiondata
mv JPEGImages JPEGImages_train
mv JPEGImages_test JPEGImages

Open tf-faster-rcnn-endernewton/experiments/scripts/test_faster_rcnn.sh
Set line 21 "ITERS = the iters of the model you trained" Say if you trained a model with 10000 iters, set this line "ITERS = 10000"

cd tf-faster-rcnn-endernewton
./experiments/scripts/test_faster_rcnn.sh 0 pascal_voc vgg16

# How to stop the training

tmux attach
ctrl+c

