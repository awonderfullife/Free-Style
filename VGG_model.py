import tensorflow as tf 
import numpy as np 
import scipy.io 

VGG_LAYERS = (
	'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

	'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

	'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 
	'rele3_3', 'conv3_4', 'relu3_4', 'pool3',

	'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 
	'rele4_3', 'conv4_4', 'relu4_4', 'pool4',

	'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 
	'rele5_3', 'conv5_4', 'relu5_4'
	)

def load_net(data_path):
	data = scipy.io.loadmat(data_path)
	if not all(i in data for i in ('layers', 'classes', 'normalization')):
		raise ValueError("Wrong model loaded!")
	mean = data['normalization'][0][0][0]
	mean_pixel = np.mean(mean, axis=(0,1))
	weights = data['layers'][0]
	return weights, mean_pixel

def net_preloaded(weight, input_image, pooling):
	net = {}
	current = input_image
	for i, name in enumerate(VGG_LAYERS):
		id_name = name[:4]
		if id_name=="conv":
			kernels, bias = weight[i][0][0][0]
			# [height, width, in_channels, out_channels] ==> [width, heright, in_channels, out_channels]
			kernels = np.transpose(kernels, (1,0,2,3)) 
			bias = bias.reshape(-1)
			current = conv_layer(current, kernels, bias)
		elif id_name=="relu":
			current = tf.nn.relu(current)
		elif id_name == "pool":
			current = pool_layer(current, pooling)
		net[name] = current
	assert len(net) == len(VGG_LAYERS)
	return net 

def conv_layer(input, weights, bias):
	conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1,1,1,1), padding="SAME")
	return tf.nn.bias_add(conv, bias)

def pool_layer(input, pooling):
	if pooling == "avg":
		return tf.nn.avg_pool(input, ksize=(1,2,2,1), strides=(1,2,2,1), padding="SAME")
	else:
		return tf.nn.max_pool(input, ksize=(1,2,2,1), strides=(1,2,2,1), padding="SAME")

def preprocess(pixels, mean_pixel):
	return pixels - mean_pixel

def unprocess(pixels, mean_pixel):
	return pixels + mean_pixel
