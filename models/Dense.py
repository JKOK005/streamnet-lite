import tensorflow as tf

class Dense(object):
	_model = None

	def __init__(self, units, activation, use_bias):
		self._model 	= tf.keras.layers.Dense( units 		= units, 
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
	def backprop_pass(self, upstream_backprop_tensor):
		(model_weights, _) 	= self._model.trainable_variables
		batch_size 			= upstream_backprop_tensor.shape[0]
		_expand_weights 	= tf.tile(tf.expand_dims(model_weights, axis=0), [batch_size,1,1])
		_expand_weights 	= tf.transpose(_expand_weights, perm = [0, 2, 1])
		return tf.matmul(upstream_backprop_tensor, _expand_weights)

	"""
	Computes dL_dw for updating weights
	"""
	def delta_weights(self, input_tensor, upstream_backprop_tensor):
		batch_size 			= upstream_backprop_tensor.shape[0]
		_expand_inputs 		= tf.transpose(input_tensor, perm = [0, 2, 1])
		batch_dw 			= tf.matmul(_expand_inputs, upstream_backprop_tensor)
		return tf.reduce_mean(batch_dw, axis = 0)

	"""
	Computes dL_db for updating weights
	"""
	def delta_bias(self, upstream_backprop_tensor):
		output_shape 	= upstream_backprop_tensor.shape
		batch_db 		= upstream_backprop_tensor * tf.ones(shape = output_shape)
		return tf.reduce_mean(batch_db, axis = 0)

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