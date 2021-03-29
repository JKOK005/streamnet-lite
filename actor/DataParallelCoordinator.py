import pykka
import logging
import uuid
import tensorflow as tf
import messages
import copy
from messages import *
from actor.Executor import StreamnetExecutor

@pykka.traversable
class DataParallelCoordinatorRoutes():
	forward_routee 	= None
	backprop_routee = None
	update_routee 	= None

	def get_forward_routee(self):
		return self.forward_routee

	def set_forward_routee(self, routee):
		self.forward_routee = routee

	def get_backprop_routee(self):
		return self.backprop_routee

	def set_backprop_routee(self, routee):
		self.backprop_routee = routee

	def get_update_routee(self):
		return self.update_routee

	def set_update_routee(self, routee):
		self.update_routee = routee

class DataParallelCoordinator(pykka.ThreadingActor):

	logger 			= logging.getLogger()
	ID 				= None
	routees 		= None
	routes 			= None

	def __init__(self, 	routees,	
						identifier = None):

		super(DataParallelCoordinator, self).__init__()
		self.ID 				= identifier if identifier else str(uuid.uuid1())
		self.routees 			= routees
		self.routes 			= DataParallelCoordinatorRoutes()
		self.streamlet_cache 	= []
		return

	def _forward_pass(self, tensor):
		self.logger.debug("ForwardPass: {0}".format(self.ID))
		split_tensors = tf.split(tensor, len(self.routees), axis = 0)
		for (each_tensor, each_routee) in zip(split_tensors, self.routees):
			each_routee.tell(ForwardStreamlet(	tensor = each_tensor, 
												fragments = len(split_tensors), 
												route_to = self.routes.get_forward_routee()
											))

	def _back_prop(self, tensor):
		self.logger.debug("Backprop: {0}".format(self.ID))
		split_tensors = tf.split(tensor, len(self.routees), axis = 0)
		for (each_tensor, each_routee) in zip(split_tensors, self.routees):
			each_routee.tell(BackpropStreamlet(	tensor = each_tensor, 
												fragments = len(split_tensors),
												route_to = self.routes.get_backprop_routee(),
												update_to = self.routes.get_update_routee()
											))

	def _clear_cache(self):
		self.streamlet_cache  = []

	def _handle_forward(self, fs):
		self.streamlet_cache.append(fs)
		if len(self.streamlet_cache) == fs.get_frag():
			tensors 	= [each_fs.get_tensor() for each_fs in self.streamlet_cache]
			full_tensor = tf.concat(tensors, axis = 0)		# Combine over batch dimensions
			self._forward_pass(tensor = full_tensor)
			self._clear_cache()

	def _handle_backprop(self, bs):
		self.streamlet_cache.append(bs)
		if len(self.streamlet_cache) == bs.get_frag():
			tensors 	= [each_fs.get_tensor() for each_fs in self.streamlet_cache]
			full_tensor = tf.concat(tensors, axis = 0)		# Combine over batch dimensions
			self._back_prop(tensor = full_tensor)
			self._clear_cache()

	def on_start(self):
		self.logger.info("Starting up Streamnet Coordinator: {0}".format(self.ID))

	def on_stop(self):
		self.logger.info("Shutting down Streamnet Coordinator: {0}".format(self.ID))

	def on_failure(self, exception_type, exception_value, traceback):
		pass

	def on_receive(self, message):
		if type(message) is messages.ScaleRoutee:
			pass
		
		elif type(message) is messages.ShrinkRoutee:
			pass

		elif type(message) is messages.ForwardStreamlet:
			self._handle_forward(fs = message)

		elif type(message) is messages.BackpropStreamlet:
			self._handle_backprop(bs = message)

