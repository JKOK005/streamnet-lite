import tensorflow as tf

class Dense(object):
	_model = None

	def __init__(self, units, activation, use_bias):
		self._model = tf.keras.layers.Dense( units 		= units, 
											 activation = activation,
											 use_bias 	= use_bias,
											)
	"""
	Computes forward pass
	"""
	def forward_pass(self, tensor):
		return self._model(tensor)

	"""
	Computes dL_dx for weights to be propagated to upstream layers
	"""
	@tf.function(experimental_relax_shapes=True)
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

	"""
	Computes dL_dw for updating weights
	"""
	@tf.function(experimental_relax_shapes=True)
	def delta_weights(self, input_tensor, dl_dy):
		with tf.GradientTape() as tape:
			[weights, _] = self._model.trainable_variables
			tape.watch(weights)
			y = self._model(input_tensor)

		size_w 	= len(weights.shape)
		size_y 	= len(y.shape[1:])
		dy_dw 	= tape.jacobian(y, weights)
		
		for _ in range(size_w):
			dl_dy = tf.expand_dims(dl_dy, -1)

		dl_dw = dy_dw * dl_dy
		return tf.math.reduce_sum(dl_dw, axis=[i for i in range(1, 1 + size_y, 1)])

	"""
	Computes dL_db for updating weights
	"""
	@tf.function(experimental_relax_shapes=True)
	def delta_bias(self, input_tensor, dl_dy):
		with tf.GradientTape() as tape:
			[_, bias] = self._model.trainable_variables
			tape.watch(bias)
			y = self._model(input_tensor)

		size_b 	= len(bias.shape)
		size_y 	= len(y.shape[1:])
		dy_db 	= tape.jacobian(y, bias)
		
		for _ in range(size_b):
			dl_dy = tf.expand_dims(dl_dy, -1)

		dl_db = dy_db * dl_dy
		return tf.math.reduce_sum(dl_db, axis=[i for i in range(1, 1 + size_y, 1)])

	"""
	Overrides existing model with new weights
	"""
	def set_params(self, weights = None, bias = None):
		(cur_weights, cur_bias) = self._model.get_weights()
		to_set_weights 			= weights if weights is not None else cur_weights
		to_set_bias 			= bias if bias is not None else cur_bias
		self._model.set_weights([to_set_weights, to_set_bias])
		return

	def get_model(self):
		return self._model