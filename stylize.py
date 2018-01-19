import VGG_model

import tensorflow as tf 
import numpy as np 

from sys import stderr
from PIL import Image
from functools import reduce

CONTENT_LAYERS = ("relu4_2", "relu5_2")
STYLE_LAYERS = ("relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1")


# This function yeild tuples (iteration, image)
def stylize(
	network,
	initial,
	initial_noiseblend,
	content,
	styles,
	preserve_colors,
	iterations,
	content_weight,
	content_weight_blend,
	style_weight,
	style_layer_weight_exp,
	style_blend_weights,
	tv_weight,
	learning_rate,
	beta1,
	beta2,
	epsilon,
	pooling,
	print_iterations=None,
	checkpoint_iterations=None):
	
	content_shape = (1,) + content.shape
	style_shapes = [(1,) + style.shape for style in styles]  # why needn't ()? Because tuple object is not callable
	content_features = {}
	style_features = [{} for item in styles]

	vgg_weights, vgg_mean_pixel = VGG_model.load_net(network)

	layer_weight = 1.0
	style_layers_weights = {}
	for style_layer in STYLE_LAYERS:
		style_layers_weights[style_layer] = layer_weight
		layer_weight *= style_layer_weight_exp

	# normalization for style layer weights
	style_layers_weights_sum = 0
	for style_layer in STYLE_LAYERS:
		style_layers_weights_sum += style_layers_weights[style_layer]
	for style_layer in STYLE_LAYERS:
		style_layers_weights[style_layer] /= style_layers_weights_sum

	# compute content features in feedforward mode
	# a context manager with it's certain area
	g = tf.Graph() 
	with g.as_default(), g.device("/cpu:0"), tf.Session() as sess:
		image = tf.placeholder('float', shape=content_shape)
		net = VGG_model.net_preloaded(vgg_weights, image, pooling)
		content_pre = np.array([VGG_model.preprocess(content, vgg_mean_pixel)])
		for layer in CONTENT_LAYERS:
			content_features[layer] = net[layer].eval(feed_dict={image: content_pre})

	# compute style features in feedforward mode
	for i in xrange(len(styles)):
		g = tf.Graph()
		with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
			image = tf.placeholder('float', shape=style_shapes[i])
			net = VGG_model.net_preloaded(vgg_weights, image, pooling)
			style_pre = np.array([VGG_model.preprocess(styles[i], vgg_mean_pixel)])
			for layer in STYLE_LAYERS:
				features = net[layer].eval(feed_dict={image: style_pre})
				features = np.reshape(features, (-1, features.shape[3]))
				gram = np.matmul(features.T, features)/features.size   # due to mutilayer, we need normalize to make combination become successfull
				style_features[i][layer] = gram

	initial_content_nosie_coeff = 1.0 - initial_noiseblend

	# make stylized iamge using backpropogation
	with tf.Graph().as_default():
		if initial is None:
			noise = np.random.normal(size=shape, scale=np.std(content)*0.1)
			initial = tf.random_normal(shape)*0.256
		else:
			initial = np.array([VGG_model.preprocess(initial, vgg_mean_pixel)])
			initial = initial.astype('float32')
			noise = np.random.normal(size=shape, scale=np.std(content)*0.1)
			initial = initial*initial_content_nosie_coeff + (tf.random_normal(shape)*0.256)*(1.0-initial_content_nosie_coeff)
		image = tf.Variable(initial)
		net = VGG_model.net_preloaded(vgg_weights, image, pooling)

		# content loss
		content_layers_weights = {}
		content_layers_weights['relu4_2'] = content_weight_blend
		content_layers_weights['relu5_2'] = 1.0 - content_weight_blend

		content_loss = 0
		content_losses = []
		for content_layer in CONTENT_LAYERS:
			content_losses.append(content_layers_weights[content_layer]*content_weight*
				(2*tf.nn.l2_loss(net[content_layer] - content_features[content_layer]) / content_features[content_layer].size))
		content_loss += reduce(tf.add, content_losses)  # map and reduce abstract function

		# style loss 
		style_loss = 0
		for i in range(len(style)):
			style_losses = []
			for style_layer in STYLE_LAYERS:
				layer = net[style_layer]
				_, height, width, number = map(lambda i: i.value, layer.get_shape())  # python: lambda para: operation_on_para 
				size = height*width*number
				feats = tf.reshape(layer, (-1, number))
				gram = tf.matmul(tf.transpose(feats), feats)/size
				style_gram = style_features[i][style_layer]
				style_losses.append(style_layers_weights[style_layer]*2*tf.nn.l2_loss(gram-style_gram)/style_gram.size)
			style_loss += style_weight*style_blend_weights[i]*reduce(tf.add, style_losses)

		










