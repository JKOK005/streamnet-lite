import tensorflow as tf

class Convolution2D(object):
	_model = None

	def __init__(self, filters, kernel, strides, use_bias):
		self._model = tf.keras.layers.Conv2D(filters = filters, kernel_size = kernel, strides = strides, 
											 activation = None, use_bias = use_bias)

	def forward_pass(self, tensor):
		return self._model(tensor)

	def backprop_pass(self, input_tensor, dl_dy):
		with tf.GradientTape() as tape:
			tape.watch(input_tensor)
			y = self._model(input_tensor)

		size_x 	= len(input_tensor.shape[1:])
		size_y  = len(y.shape[1:])
		dy_dx = tape.batch_jacobian(y, input_tensor)

		for _ in range(size_x):
			dl_dy = tf.expand_dims(dl_dy, -1)
		
		dl_dx = dy_dx * dl_dy
		return tf.math.reduce_sum(dl_dx, axis=[i for i in range(1, 1 + size_y, 1)])

	# Input is of shape 	[batch, in_height, in_width, in_channels]
	# Backprop is of shape 	[batch, out_height, out_width, filters]
	def delta_weights(self, input_tensor, dl_dy):
		with tf.GradientTape() as tape:
			[weights, _] = self._model.trainable_variables
			tape.watch(weights)
			y = self._model(input_tensor)

		size_w 		= len(weights.shape)
		size_y 		= len(y.shape[1:])
		dy_dw 		= tape.jacobian(y, weights)
		
		for _ in range(size_w):
			dl_dy = tf.expand_dims(dl_dy, -1)

		dl_dw = dy_dw * dl_dy
		return tf.math.reduce_sum(dl_dw, axis=[i for i in range(1, 1 + size_y, 1)])

	def delta_bias(self, dl_dy):
		with tf.GradientTape() as tape:
			[_, bias] = self._model.trainable_variables
			tape.watch(bias)
			y = self._model(input_tensor)

		size_b 		= len(bias.shape)
		size_y 		= len(y.shape[1:])
		dy_db 		= tape.jacobian(y, bias)
		
		for _ in range(size_b):
			dl_dy = tf.expand_dims(dl_dy, -1)

		dl_db = dy_db * dl_dy
		return tf.math.reduce_sum(dl_db, axis=[i for i in range(1, 1 + size_y, 1)])

	def set_params(self, weights = None, bias = None):
		pass

	def get_model(self):
		return self._model