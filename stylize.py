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

	