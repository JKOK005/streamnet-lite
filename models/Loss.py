import tensorflow as tf

class Loss(object):
	tf_loss 	= None

	def __init__(self, tf_loss):
		self.tf_loss = tf_loss

	def compute(self, tensor, label):
		with tf.GradientTape() as tape:
			tape.watch([tensor])
			cost = self.tf_loss(label, tensor)
		grad = tape.gradient(cost, tensor)
		return (grad, cost)