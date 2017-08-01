# _______________________________________________________
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# Modified by Yunqiu Xu
# --------------------------------------------------------


# A revision of VGG16 (tensorflow backend)
# Input : 224 * 224 * 3
# (after) conv1 : 224 * 224 * 64
# maxpool : 112 * 112 * 64
# conv2 : 112 * 112 * 128
# maxpool : 56 * 56 * 128
# conv3 : 56 * 56 * 256
# maxpool : 28 * 28 * 256
# conv4 : 28 * 28 * 512
# maxpool : 14 * 14 * 512
# conv5 : 14 * 14 * 512
# maxpool : 7 * 7 * 512
# fc6 : 4096
# fc7 : 4096
# ---------------------------------------------------------


# Why Faster RCNN is faster : RPN
#   RCNN: 
#      get proposal --> get features(CNN) --> SVM --> bbox regression
#   Fast RCNN: 
#      send proposal and features to ROI pooling --> combine bbox and SVM together
#   Faster RCNN:
#      get features first --> get proposals from RPN --> send proposal and features to ROI pooling
# ---------------------------------------------------------


# ------------- To do 1: Multiple Scale Faster-RCNN ------
# Combine both global and local features --> enhance hand detecting in an image
# Collect features not only conv5, but also conv3 and conv4, then incorporate them
# Implementation: 
#   1. For conv3, conv4, conv5, each conv is only followed with ReLU, remove Max-pooling layer.
#   2. Take their output as the input of 3 corresponding ROI pooling layers and normalization layers
#   3. Concat and shrink normalization layers as input of fc layers
#   4. roi pooling in fc layers: make prediction of class and position
# --------------------------------------------------------
# ------------- To do 2: Weight Normalization ------------
# Features in shallower layers: larger-scaled values
# Features in deeper layers: smaller-scaled values
# To combine the features of 3 conv layers, we need to normalize them
# Implementation:
#   1. Put each feature into normalization layer(see the equations)
#   2. Each pixel xi is normalized, then multiply scaling factor ri
#   3. Use backpropagation to get ri in training step, we need to build loop here
#   4. After normalization, the features will be concated
# --------------------------------------------------------
# ------------- To do 3 Add New Layer --------------------
# 1. Each RPN needs a normalization layer
# 2. Add two more ROI pooling layers in detector part
# 3. Each ROI pooling layer needs a normalization layer
# 4. After each concatenation(2 positions in total), we need a 1*1 conv layer
# --------------------------------------------------------


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
import numpy as np

from nets.network import Network
from model.config import cfg

class vgg16(Network):
  def __init__(self, batch_size=1):
    Network.__init__(self, batch_size=batch_size)
    self._arch = 'vgg16'

  # [Hand Detection] Batch normalization
  # http://stackoverflow.com/a/34634291/2267819
  # Note that this is different from the paper(they use another method)
  def batch_norm_layer(self, to_be_normalized, is_training):
    if is_training:
      train_phase = tf.constant(1)
    else:
      train_phase = tf.constant(-1)
    beta = tf.Variable(tf.constant(0.0, shape=[to_be_normalized.shape[-1]]), name='beta', trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[to_be_normalized.shape[-1]]), name='gamma', trainable=True)
    axises = np.arange(len(to_be_normalized.shape) - 1)
    batch_mean, batch_var = tf.nn.moments(to_be_normalized, axises, name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.5)

    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(train_phase > 0, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var))) # if is training --> update
    normed = tf.nn.batch_normalization(to_be_normalized, mean, var, beta, gamma, 1e-3)
    return normed


  def build_network(self, sess, is_training=True):
    with tf.variable_scope('vgg_16', 'vgg_16'):
      # select initializers
      if cfg.TRAIN.TRUNCATED:
        initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
        initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
      else:
        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)

      # [VGG16] conv1
      # input shape : 224 * 224 * 3
      # output shape : 112 * 112 * 64
      net = slim.repeat(self._image, 2, slim.conv2d, 64, [3, 3],
                        trainable=False, scope='conv1')
      net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool1')

      # [VGG16] conv2
      # input shape : 112 * 112 * 64
      # output shape : 56 * 56 * 128
      net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3],
                        trainable=False, scope='conv2')
      net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool2')


      # [Hand Detection] REMOVE net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool3') 
      # [Hand Detection] conv3
      # input shape : 56 * 56 * 128
      # output shape : 56 * 56 * 256
      net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3],
                        trainable=is_training, scope='conv3')
      to_be_normalized_1 = net 
      # [Hand Detection] conv4
      # input shape : 56 * 56 * 256
      # output shape : 56 * 56 * 256
      net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3],
                        trainable=is_training, scope='conv4')
      to_be_normalized_2 = net 
      # [Hand Detection] conv5
      # input shape : 56 * 56 * 256
      # output shape : 56 * 56 * 256
      net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3],
                        trainable=is_training, scope='conv5')
      to_be_normalized_3 = net 

# ------------- Take a break -----------------------------
# Now as we get to_be_normalized_1 / to_be_normalized_2 / to_be_normalized_3, each is 56 * 56 * 256
# For RPN , we need to: 
# 1. normalize each to_be_normalized layer
# 2. concat 3 normalized layers
# 3. change the dimension using 1 * 1 conv
# 3. Then the modified net can be used in RPN
# 
# For ROI pooling, we need to:
# 1. put each conv output into its ROI pooling (so there should be 3 ROI pooling layers)
# 2. normalize each layer
# 3. concat them
# 4. change the dimension using 1 * 1 conv
# --------------------------------------------------------

      # ------------- Normalization for RPN --------------------
      # old version 
      # normed_1_rpn = tf.nn.l2_normalize(to_be_normalized_1, dim = [0, 1])
      # normed_2_rpn = tf.nn.l2_normalize(to_be_normalized_2, dim = [0, 1])
      # normed_3_rpn = tf.nn.l2_normalize(to_be_normalized_3, dim = [0, 1])
      normed_1_rpn = self.batch_norm_layer(to_be_normalized_1, is_training)
      normed_2_rpn = self.batch_norm_layer(to_be_normalized_2, is_training)
      normed_3_rpn = self.batch_norm_layer(to_be_normalized_3, is_training)
      
      # ------------- Concatation for RPN (56 * 56 * 768) ------
      # old version
      # concated_rpn = tf.concat([normed_1_rpn, normed_2_rpn, normed_3_rpn], 2)
      #batch *length*width*channel
      #concate in the channel
      concated_rpn = tf.concat([normed_1_rpn, normed_2_rpn, normed_3_rpn], -1)
     
      # ------------- 1 * 1 conv -------------------------------
      scaled_rpn = slim.conv2d(concated_rpn, 512, [1, 1], trainable=is_training, weights_initializer=initializer, scope="scaled_rpn/1x1")
      # Then we can get 56 * 56 * 512
      
      
      # [Faster RCNN] summary and anchor
      self._act_summaries.append(scaled_rpn)
      self._layers['head'] = scaled_rpn
      self._anchor_component()

      # ------------- RPN Begin --------------------------------
      
      rpn = slim.conv2d(scaled_rpn, 512, [3, 3], trainable=is_training, weights_initializer=initializer, scope="rpn_conv/3x3")
      self.show_variables("rpn",rpn.get_shape())

      print("rpn",rpn.get_shape())
      self._act_summaries.append(rpn)
      rpn_cls_score = slim.conv2d(rpn, self._num_anchors * 2, [1, 1], trainable=is_training,
                                  weights_initializer=initializer,
                                  padding='VALID', activation_fn=None, scope='rpn_cls_score')
      # [Hand Detection] change it so that the score has 2 as its channel size
      rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2, 'rpn_cls_score_reshape')
      rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape")
      rpn_cls_prob = self._reshape_layer(rpn_cls_prob_reshape, self._num_anchors * 2, "rpn_cls_prob")
      rpn_bbox_pred = slim.conv2d(rpn, self._num_anchors * 4, [1, 1], trainable=is_training,
                                  weights_initializer=initializer,
                                  padding='VALID', activation_fn=None, scope='rpn_bbox_pred')
      print("rpn_cls_score",rpn_cls_score.get_shape())
      if is_training:
        print("Compute rois,roi_scores")
        print("training:rpn_cls_score",rpn_cls_score.get_shape())
        rois, roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
    
        print("Compute rpn_labels")
        self.show_variables("rpn_cls_score",rpn_cls_score.get_shape())

        rpn_labels = self._anchor_target_layer(rpn_cls_score, "anchor")
        # Try to have a determinestic order for the computing graph, for reproducibility
        with tf.control_dependencies([rpn_labels]):
          rois, _ = self._proposal_target_layer(rois, roi_scores, "rpn_rois")
      else:
        if cfg.TEST.MODE == 'nms':
          rois, _ = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
        elif cfg.TEST.MODE == 'top':
          rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
        else:
          raise NotImplementedError
      # ------------- RPN End ----------------------------------

      print("vgg16_rois",str(rois.shape))

      # ------------- ROI Pooling Begin ------------------------
      if cfg.POOLING_MODE == 'crop':
        # get roi layers
        roi1 = self._crop_pool_layer(to_be_normalized_1, rois, "roi1") # 28 * 28 * 256
        #print("vgg16_roi1",str(roi1.shape))
        roi2 = self._crop_pool_layer(to_be_normalized_2, rois, "roi2") # 28 * 28 * 256
        roi3 = self._crop_pool_layer(to_be_normalized_3, rois, "roi3") # 28 * 28 * 256
        # normalization
        normed_1_roi = self.batch_norm_layer(roi1, is_training)
        normed_2_roi = self.batch_norm_layer(roi2, is_training)
        normed_3_roi = self.batch_norm_layer(roi3, is_training)
        # concat
        concated_roi = tf.concat([normed_1_roi, normed_2_roi, normed_3_roi], -1) # 28 * 28 * 768
       
        #concated_roi = tf.slice(concated_roi,[0,0,0,0],[channel1,-1,-1,-1])#train 256 testing 300
        #print("concated_roi",concated_roi.get_shape())
        
      # scale
        #with tf.variable_scope("rois") as scope:
        #  out = rois.shape[0]

        pool5 = slim.conv2d(concated_roi,512, [1, 1], trainable=is_training, weights_initializer=initializer, scope="pool5/1x1") # 28 * 28 * 512

        #print("pool5",pool5.get_shape())
        #pool5 = tf.reshape(pool5,[-1,])
        #pool5 = tf.slice(pool5,[0,0,0,0],[self._anchor_length,-1,-1,-1])

      else:
        raise NotImplementedError
      # old version
      # if cfg.POOLING_MODE == 'crop':
      #  roi_pool_1 = self._crop_pool_layer(to_be_normalized_1, rois, "roi_pool_1")
      #  roi_pool_2 = self._crop_pool_layer(to_be_normalized_2, rois, "roi_pool_2")
      #  roi_pool_3 = self._crop_pool_layer(to_be_normalized_3, rois, "roi_pool_3")

      #  roi_pool_1_normalized = tf.nn.l2_normalize(roi_pool_1, dim = [0, 1])
      #  roi_pool_2_normalized = tf.nn.l2_normalize(roi_pool_2, dim = [0, 1])
      #  roi_pool_3_normalized = tf.nn.l2_normalize(roi_pool_3, dim = [0, 1])
      #  pool5 = tf.concat([roi_pool_1_normalized, roi_pool_1_normalized, roi_pool_1_normalized], 2)
      # ------------- ROI Pooling End --------------------------


      # [VGG16] flatten
      pool5_flat = slim.flatten(pool5, scope='flatten')
      # [VGG16] dense 4096 + dropout
      fc6 = slim.fully_connected(pool5_flat, 4096, scope='fc6')
      if is_training:
        fc6 = slim.dropout(fc6, keep_prob=0.5, is_training=True, scope='dropout6')
      # [VGG16] dense 4096 + dropout
      fc7 = slim.fully_connected(fc6, 4096, scope='fc7')
      if is_training:
        fc7 = slim.dropout(fc7, keep_prob=0.5, is_training=True, scope='dropout7')
      
      # [Faster RCNN] get cls_score(class) and bbox_predict(position)
      cls_score = slim.fully_connected(fc7, self._num_classes, 
                                       weights_initializer=initializer,
                                       trainable=is_training,
                                       activation_fn=None, scope='cls_score')
      cls_prob = self._softmax_layer(cls_score, "cls_prob")
      bbox_pred = slim.fully_connected(fc7, self._num_classes * 4, 
                                       weights_initializer=initializer_bbox,
                                       trainable=is_training,
                                       activation_fn=None, scope='bbox_pred')
      #print("cls_score",cls_score.get_shape())
      #if not is_training and len(rois.shape)==2:
      #  bbox_pred = tf.slice(bbox_pred,[0,0,0,0],[rois.shape[0],-1,-1,-1])

      print("vgg16_bbox_pred",str(bbox_pred.shape))
      print("vgg16_rois",str(rois.shape))  
   
      self._predictions["rpn_cls_score"] = rpn_cls_score
      self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
      self._predictions["rpn_cls_prob"] = rpn_cls_prob
      self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
      self._predictions["cls_score"] = cls_score
      self._predictions["cls_prob"] = cls_prob
      self._predictions["bbox_pred"] = bbox_pred
      self._predictions["rois"] = rois # not original rois

      self._score_summaries.update(self._predictions)

      return rois, cls_prob, bbox_pred


  def get_variables_to_restore(self, variables, var_keep_dic):
    variables_to_restore = []
    #[Hand Detection]
    var_modified=['vgg_16/conv3/conv3_1/weights:0','vgg_16/conv3/conv3_2/weights:0','vgg_16/conv3/conv3_3/weights:0',
                      'vgg_16/conv4/conv4_1/weights:0','vgg_16/conv4/conv4_2/weights:0','vgg_16/conv4/conv4_3/weights:0',
                      'vgg_16/conv5/conv5_1/weights:0','vgg_16/conv5/conv5_2/weights:0','vgg_16/conv5/conv5_3/weights:0',
                      'vgg_16/conv3/conv3_1/biases:0','vgg_16/conv3/conv3_2/biases:0','vgg_16/conv3/conv3_3/biases:0',
                      'vgg_16/conv4/conv4_1/biases:0','vgg_16/conv4/conv4_2/biases:0','vgg_16/conv4/conv4_3/biases:0',
                      'vgg_16/conv5/conv5_1/biases:0','vgg_16/conv5/conv5_2/biases:0','vgg_16/conv5/conv5_3/biases:0']
    #/[Hand Detection]

    for v in variables:
      # exclude the conv weights that are fc weights in vgg16
      if v.name == 'vgg_16/fc6/weights:0' or v.name == 'vgg_16/fc7/weights:0':
        self._variables_to_fix[v.name] = v
        continue
      # exclude the first conv layer to swap RGB to BGR
      if v.name == 'vgg_16/conv1/conv1_1/weights:0':
        self._variables_to_fix[v.name] = v
        continue

      # [Hand Detection]
      if v.name in var_modified:
          continue
      # /[Hand Detection]

      if v.name.split(':')[0] in var_keep_dic:
        print('Varibles restored: %s' % v.name)
        variables_to_restore.append(v)

    return variables_to_restore

  def fix_variables(self, sess, pretrained_model):
    print('Fix VGG16 layers..')
    with tf.variable_scope('Fix_VGG16') as scope:
      with tf.device("/cpu:0"):
        # fix the vgg16 issue from conv weights to fc weights
        # fix RGB to BGR
        #fc6_conv = tf.get_variable("fc6_conv", [7, 7, 512, 4096], trainable=False)
        #fc7_conv = tf.get_variable("fc7_conv", [1, 1, 4096, 4096], trainable=False)
        conv1_rgb = tf.get_variable("conv1_rgb", [3, 3, 3, 64], trainable=False)
        restorer_fc = tf.train.Saver({ "vgg_16/conv1/conv1_1/weights": conv1_rgb})
        restorer_fc.restore(sess, pretrained_model)

        #sess.run(tf.assign(self._variables_to_fix['vgg_16/fc6/weights:0'], tf.reshape(fc6_conv, 
        #                    self._variables_to_fix['vgg_16/fc6/weights:0'].get_shape())))
        #sess.run(tf.assign(self._variables_to_fix['vgg_16/fc7/weights:0'], tf.reshape(fc7_conv, 
        #                    self._variables_to_fix['vgg_16/fc7/weights:0'].get_shape())))
        sess.run(tf.assign(self._variables_to_fix['vgg_16/conv1/conv1_1/weights:0'],tf.reverse(conv1_rgb, [2])))
  
  def show_variables(self,var_name,var):
    print(var_name,var)
