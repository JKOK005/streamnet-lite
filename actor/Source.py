import pykka
import logging
import messages
from messages import *
import tensorflow as tf
import time

class StreamnetSource(pykka.ThreadingActor):

	logger 	= logging.getLogger()
	ID 		= None

	def __init__(self, identifier: str = None):
		super().__init__()
		self.ID = identifier
		return

	def on_start(self):
		self.logger.info("Starting up Streamnet Executor")

	def on_stop(self):
		self.logger.info("Shutting down Streamnet Executor")

	def on_failure(self, exception_type, exception_value, traceback):
		pass		

	def on_receive(self, message):	
		if 	type(message) is messages.StartSourceStreamlet:
			fwd_tensor 		= tf.random.uniform(shape = [4,1,5])
			self.logger.info("Received call to start")
			
			while(True):
				fwd_stream 	= ForwardStreamlet(tensor = fwd_tensor, fragments = 1)
				message.get_route_to().tell(fwd_stream)
				time.sleep(message.get_interval())