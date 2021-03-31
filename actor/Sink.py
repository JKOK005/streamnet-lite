import pykka
import logging
import tensorflow as tf
import time
import messages
from messages import *
from utils.StreamletTools import StreamletTools

class StreamnetSink(pykka.ThreadingActor):

	logger 				= logging.getLogger()
	ID 					= None
	loss_model 			= None
	backward_routee		= None
	streamlet_cache 	= None
	received_pred 		= None
	received_label 		= None

	def __init__(self, 	backward_routee, 
						loss_model,
						identifier: str = None):
		super().__init__()
		self.ID 				= identifier
		self.loss_model 		= loss_model
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

	def reset(self):
		self._clear_cache()
		self.received_pred 	= None
		self.received_label 	= None

	def compute_loss(self):
		# Check if all data is present before computing loss
		if self.received_pred is not None and self.received_label is not None:
			(prediction, truth, sort_index) = StreamletTools.join_on_index(streamlet_A = self.received_pred, streamlet_B = self.received_label)
			# TODO: Loss computed is overall loss and not derivative of loss to each unit of the output.
			# 		This needs to be addressed
			
			computed_loss 	= self.loss_model(truth, prediction)
			bp_stream 		= BackpropStreamlet(tensor = computed_loss, fragments = 1, index = sort_index)
			self.backward_routee.tell(bp_stream)
			self.reset()
		return

	def on_receive(self, message):		
		if 	type(message) is messages.ForwardStreamlet:
			self.logger.debug("Received forward streamlet in sink of shape: {0}".format(message.get_tensor().shape))
			self.streamlet_cache.append(message)
			if len(self.streamlet_cache) == message.get_frag():
				self.received_pred = StreamletTools.merge(streamlets = self.streamlet_cache)

		elif type(message) is messages.LabelStreamlet:
			self.logger.debug("Received label streamlet in sink of shape: {0}".format(message.get_tensor().shape))
			self.received_label = message

		self.compute_loss()
		