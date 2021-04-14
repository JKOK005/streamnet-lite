import pykka
import tensorflow as tf
import messages
from messages import *

class StreamnetFlatten(pykka.ThreadingActor):
	inpt_shape = None

	def __init__(self):
		super().__init__()
		self._model = tf.keras.layers.Flatten(data_format = "channels_last")

	def forward_pass(self, tensor):
		self.inpt_shape = tensor.shape
		return self._model(tensor)

	def backprop_pass(self, tensor):
		return tf.reshape(tensor = tensor, shape = self.inpt_shape)

	def on_receive(self, message):
		tensor 			= message.get_tensor()

		if type(message) is messages.ForwardStreamlet:
			fwd_tensor 		= self.forward_pass(tensor = tensor)
			fwd_stream 		= ForwardStreamlet(tensor = fwd_tensor, fragments = message.get_frag(), index = message.get_index())
			forward_routee 	= message.get_route_to()
			forward_routee.tell(fwd_stream)

		elif type(message) is messages.BackpropStreamlet:
			bp_tensor 		= self.backprop_pass(tensor = tensor)
			bp_stream 		= BackpropStreamlet(tensor = bp_tensor, fragments = message.get_frag(), index = message.get_index())
			bp_routee 		= message.get_route_to()
			bp_routee.tell(bp_stream)