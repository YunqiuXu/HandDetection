#!/bin/bash

GPU_ID=$1
DATASET="pascal_voc"
NET=$2

case ${NET} in
  vgg16)
    mv lib/nets/network_vgg16.py lib/nets/network.py
    bash experiments/scripts/test_faster_rcnn.sh $GPU_ID $DATASET $NET
    mv lib/nets/network.py lib/nets/network_vgg16.py 
    ;;
  res101)
    mv lib/nets/network_resnet.py lib/nets/network.py
    bash experiments/scripts/test_faster_rcnn.sh $GPU_ID $DATASET $NET
    mv lib/nets/network.py lib/nets/network_resnet.py
    ;;
  *)
    echo "You should choose either vgg16 or res101"
    exit
    ;;
esac
