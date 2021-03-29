import pykka
import logging
import tensorflow as tf
import time
import messages
from messages import *

class StreamnetSink(pykka.ThreadingActor):

	logger 				= logging.getLogger()
	ID 					= None
	backward_routee		= None

	def __init__(self, 	backward_routee, 
						loss,
						identifier: str = None):
		super().__init__()
		self.ID 				= identifier
		self.backward_routee 	= backward_routee
		self.streamlet_cache 	= []
		return

	def _clear_cache(self):
		self.streamlet_cache  = []

	def on_start(self):
		self.logger.info("Starting up Streamnet Executor")

	def on_stop(self):
		self.logger.info("Shutting down Streamnet Executor")

	def on_failure(self, exception_type, exception_value, traceback):
		pass

	def on_receive(self, message):		
		if 	type(message) is messages.ForwardStreamlet:
			self.logger.debug("Received forward streamlet in sink of shape: {0}".format(message.get_tensor().shape))
			self.streamlet_cache.append(message)
			if len(self.streamlet_cache) == message.get_frag():
				# TODO: Replace with loss logic
				bp_tensor 	= tf.random.uniform(shape=[2**15,1,128])
				bp_stream 	= BackpropStreamlet(tensor = bp_tensor, fragments = 1)
				self.backward_routee.tell(bp_stream)
				self._clear_cache()
