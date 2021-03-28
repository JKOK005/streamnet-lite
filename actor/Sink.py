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
	streamlet_cache 	= []

	def __init__(self, 	backward_routee, 
						identifier: str = None):
		super().__init__()
		self.ID 				= identifier
		self.backward_routee 	= backward_routee
		return

	def on_start(self):
		self.logger.info("Starting up Streamnet Executor")

	def on_stop(self):
		self.logger.info("Shutting down Streamnet Executor")

	def on_failure(self, exception_type, exception_value, traceback):
		pass

	def on_receive(self, message):		
		if 	type(message) is messages.ForwardStreamlet:
			self.logger.info("Received forward streamlet in sink")
			self.streamlet_cache.append(message)
			if len(self.streamlet_cache) == message.get_frag():
				# TODO: Replace with loss logic
				bp_tensor 	= tf.random.uniform(shape=[4,1,3])
				bp_stream 	= BackpropStreamlet(tensor = bp_tensor, fragments = 1)
				self.backward_routee.tell(bp_stream)
