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

	def _clear_cache(self):
		self.weights_cache 	= []
		self.bias_cache 	= []

	def _reduce_msg(self):
		if len(self.weights_cache) == self.weights_cache[0].get_frag() and len(self.bias_cache) == self.bias_cache[0].get_frag():
			weight_streamlet = StreamletTools.reduce_on_batch(streamlets = self.weights_cache)
			bias_streamlet 	 = StreamletTools.reduce_on_batch(streamlets = self.bias_cache)
			self.logger.debug(f"Reduced weights of shape: {weight_streamlet.get_tensor().shape} and bias of shape: {bias_streamlet.get_tensor().shape}")
			streamlet = WeightBiasStreamlet(reduced_weight_streamlet = weight_streamlet, reduced_bias_streamlet = bias_streamlet)
			self.layer_coordinator.tell(streamlet)
			self._clear_cache()

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
			self.weights_cache.append(message)
			self._reduce_msg()
		
		elif type(message) is messages.BiasStreamlet:
			self.bias_cache.append(message)
			self._reduce_msg()