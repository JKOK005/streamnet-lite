import pykka
import logging
import messages
import uuid
from messages import *
from utils.StreamletTools import StreamletTools

class StreamnetOptimizer(pykka.ThreadingActor):

	logger 			= logging.getLogger()

	def __init__(self, 	layer_coordinator, 
						identifier: str = None):

		super().__init__()
		self.ID 				= identifier if identifier else str(uuid.uuid1())
		self.layer_coordinator 	= layer_coordinator
		self.weights_cache 		= []
		self.bias_cache 		= []
		return

	def _handle_weights(self, ws):
		self.weights_cache.append(ws)
		if len(self.weights_cache) == ws.get_frag():
			streamlet = StreamletTools.reduce_on_batch(streamlets = self.weights_cache)
			self.logger.debug(f"Reduced weights of shape: {streamlet.get_tensor().shape}")
			self.layer_coordinator.tell(streamlet)

	def _handle_bias(self, bs):
		self.bias_cache.append(bs)
		if len(self.bias_cache) == bs.get_frag():
			streamlet = StreamletTools.reduce_on_batch(streamlets = self.bias_cache)
			self.logger.debug(f"Reduced bias of shape: {streamlet.get_tensor().shape}")
			self.layer_coordinator.tell(streamlet)

	def on_start(self):
		self.logger.info("Starting up Streamnet Executor")

	def on_stop(self):
		self.logger.info("Shutting down Streamnet Executor")

	def on_failure(self, exception_type, exception_value, traceback):
		pass

	def on_receive(self, message):
		tensor = message.get_tensor()
		self.logger.debug("Received tensor of shape: {0}".format(tensor.shape))

		if type(message) is messages.WeightStreamlet:
			self._handle_weights(ws = message)
		
		elif type(message) is messages.BiasStreamlet:
			self._handle_bias(bs = message)