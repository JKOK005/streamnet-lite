import tensorflow as tf

class Loss(object):
	tf_loss 	= None

	def __init__(self, tf_loss):
		self.tf_loss = tf_loss

	def _wrap_loss_fnct(self, label):
		def fnct(pred):
			return self.tf_loss(label, pred)
		return fnct

	def compute(self, tensor, label):
		with tf.GradientTape() as tape:
			tape.watch([tensor])
			cost = self.tf_loss(tensor, label)
		grad = tape.gradient(cost, tensor)
		return grad