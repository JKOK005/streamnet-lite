import pykka
import logging
import tensorflow as tf
import messages
import copy
from messages import *
from utils.StreamletTools import StreamletTools

class StreamnetActivation(pykka.ThreadingActor):

	logger = logging.getLogger()

	def __init__(self, 	activation, 
						identifier: str = None):

		super().__init__()
		self.ID 			= identifier
		self.activation 	= activation
		self.cur_input 		= None
		return

	def on_start(self):
		self.logger.info("Starting up Streamnet Activation")

	def on_stop(self):
		self.logger.info("Shutting down Streamnet Activation")

	def on_failure(self, exception_type, exception_value, traceback):
		pass

	def on_receive(self, message):
		tensor = message.get_tensor()
		self.logger.debug("Received tensor of shape: {0}".format(tensor.shape))

		if 	type(message) is messages.ForwardStreamlet:
			fwd_tensor 		= self.activation.forward_pass(tensor = tensor)
			fwd_stream 		= ForwardStreamlet(tensor = fwd_tensor, fragments = message.get_frag(), index = message.get_index())
			self.cur_input 	= tensor
			forward_routee 	= message.get_route_to()
			forward_routee.tell(fwd_stream)

		elif type(message) is messages.BackpropStreamlet:
			bp_tensor 		= self.activation.backprop_pass(input_tensor = self.cur_input, upstream_backprop_tensor = tensor)
			bp_stream 		= BackpropStreamlet(tensor = bp_tensor, fragments = message.get_frag(), index = message.get_index())
			bp_routee 		= message.get_route_to()		
			bp_routee.tell(bp_stream)