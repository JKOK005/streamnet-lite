import tensorflow as tf

class Sigmoid(object):
	def __init__(self):
		super(Sigmoid, self).__init__()

	def forward_pass(self, tensor):
		return tf.sigmoid(tensor)

	def backprop_pass(self, input_tensor, upstream_backprop_tensor):
		inpt_sigmoid = tf.sigmoid(input_tensor)
		return inpt_sigmoid * (1 - inpt_sigmoid) * upstream_backprop_tensor