import pykka
import logging
from messages import *
import tensorflow as tf
import time

class StreamnetSource(pykka.ThreadingActor):

	logger 			= logging.getLogger()
	ID 				= None
	forward_routee	= None

	def __init__(self, 	forward_routee, 
						identifier: str = None):

		super().__init__()
		self.ID 				= identifier
		self.forward_routee 	= forward_routee
		return

	def on_start(self):
		logging("Starting up Streamnet Executor")

	def on_stop(self):
		logging("Shutting down Streamnet Executor")

	def on_failure(self, exception_type, exception_value, traceback):
		pass

	def initiate(self, interval):
		fwd_tensor 	= tf.random.uniform(shape = [4,1,5])
		fwd_stream 	= ForwardStreamlet(tensor = fwd_tensor, fragments = 1)
		self.forward_routee.tell(fwd_stream)
		time.sleep(interval)

	def on_receive(self, message):		
		if 	message is BackpropStreamlet:
			self.logger.info("Received back prop back to source")
