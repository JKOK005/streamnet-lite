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
		self.weight_streamlet 	= None
		self.bias_streamlet 	= None
		return

	def _reset(self):
		self.weights_cache 		= []
		self.bias_cache 		= []
		self.weight_streamlet 	= None
		self.bias_streamlet 	= None

	def _attempt_dispatch(self):
		if self.weight_streamlet is not None and self.bias_streamlet is not None:
			streamlet = WeightBiasStreamlet(reduced_weight_streamlet = self.weight_streamlet, reduced_bias_streamlet = self.bias_streamlet)
			self.layer_coordinator.tell(streamlet)
			self._reset()

	def _reduce_weight(self):
		if len(self.weights_cache) == self.weights_cache[0].get_frag():
			import time
			start = time.time()
			self.weight_streamlet = StreamletTools.reduce_on_batch(streamlets = self.weights_cache)
			self.logger.info("Reduce weights: {0}".format(time.time() - start))
			self._attempt_dispatch()

	def _reduce_bias(self):
		if len(self.bias_cache) == self.bias_cache[0].get_frag():
			import time
			start = time.time()
			self.bias_streamlet = StreamletTools.reduce_on_batch(streamlets = self.bias_cache)
			self.logger.info("Reduce bias: {0}".format(time.time() - start))
			self._attempt_dispatch()

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
			self._reduce_weight()
		
		elif type(message) is messages.BiasStreamlet:
			self.bias_cache.append(message)
			self._reduce_bias()