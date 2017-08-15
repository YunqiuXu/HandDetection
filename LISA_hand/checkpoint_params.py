# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 17:00:50 2017

@author: Shaoshen Wang
"""
#Used for show the variables in a checkpoint file
#Usage: Put this code under tf-faster-rcnn-master

import os
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

def get_variables_in_checkpoint_file(file_name):
    try:
        reader = pywrap_tensorflow.NewCheckpointReader(file_name)      
        var_to_shape_map = reader.get_variable_to_shape_map()
        return var_to_shape_map
    except Exception as e:
        print(str(e))


model_dir=".\data\imagenet_weights"
checkpoint_path = os.path.join(model_dir, "vgg16.ckpt")

#print(type(file_name))

var_to_shape_map=get_variables_in_checkpoint_file(checkpoint_path)

for var in var_to_shape_map:
    print(var,var_to_shape_map[var])


# List ALL tensors example output: v0/Adam (DT_FLOAT) [3,3,1,80]
#print_tensors_in_checkpoint_file(file_name=checkpoint_path, tensor_name='',all_tensors='')
