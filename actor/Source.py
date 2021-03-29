import pykka
import logging
import messages
from messages import *
import tensorflow as tf
import time

class StreamnetSource(pykka.ThreadingActor):

	logger 				= logging.getLogger()
	ID 					= None
	epoch_start_time 	= 0

	def __init__(self, identifier: str = None):
		super().__init__()
		self.ID = identifier
		self.streamlet_cache = []
		return

	def _clear_cache(self):
		self.streamlet_cache  = []

	def on_start(self):
		self.logger.info("Starting up Streamnet Executor")

	def on_stop(self):
		self.logger.info("Shutting down Streamnet Executor")

	def on_failure(self, exception_type, exception_value, traceback):
		pass		

	def start_ingest(self):
		self.epoch_start_time = time.time()
		fwd_tensor 	= tf.random.uniform(shape = [2**15,1,20])
		fwd_stream 	= ForwardStreamlet(tensor = fwd_tensor, fragments = 1)
		self.route_to.tell(fwd_stream)

	def on_receive(self, message):	
		if 	type(message) is messages.StartSourceStreamlet:
			self.route_to 	= message.get_route_to()
			self.start_ingest()

		elif type(message) is messages.BackpropStreamlet:
			self.streamlet_cache.append(message)
			if len(self.streamlet_cache) == message.get_frag():
				self.logger.info("Time taken: {0} s".format(time.time() - self.epoch_start_time))
				self._clear_cache()
				self.start_ingest() 	# Repeat ingestion process
