import tensorflow as tf

class Loss(object):
	tf_loss 	= None

	def __init__(self, tf_loss):
		self.tf_loss = tf_loss

	def compute(self, tensor, label):
		return self.tf_loss(tensor, label)