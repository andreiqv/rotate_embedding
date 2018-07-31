#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
np.set_printoptions(precision=4, suppress=True)

import tensorflow as tf
import tensorflow_hub as hub

def calculate_embedding(images, shape):

	height, width, color =  shape
	bottleneck_tensor_size = 1001

	x = tf.placeholder(tf.float32, [None, height, width, 3], name='Placeholder-x')

	resized_input_tensor = tf.reshape(x, [-1, height, width, 3])

	module = hub.Module("https://tfhub.dev/google/imagenet/resnet_v2_152/classification/1")	
	assert height, width == hub.get_expected_image_size(module)
	
	bottleneck_tensor = module(resized_input_tensor)  # Features with shape [batch_size, num_features]
	print('bottleneck_tensor:', bottleneck_tensor)

	with tf.Session() as sess:  # Connect to the TF runtime.
		init = tf.global_variables_initializer()
		sess.run(init)	# Randomly initialize weights.
		
		embedding = [bottleneck_tensor.eval(\
					feed_dict={ x : images[i:i+1] })\
					for i in range(len(images))]	

	print(len(embedding))				
	return embedding
	