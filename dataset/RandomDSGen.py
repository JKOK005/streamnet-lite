import tensorflow as tf
import numpy as np

class RandomDSGen(object):
	def __init__(self, input_shape, label_shape):
		self.input_shape = input_shape
		self.label_shape = label_shape

	def gen_ds(self, batch_size: int):
		while True:
			inpt 	= tf.random.uniform(shape = [batch_size] + self.input_shape)
			labels 	= tf.random.uniform(shape = [batch_size] + self.label_shape)
			yield(inpt, labels)