import pykka
import logging
import uuid
import tensorflow as tf
import messages
import copy
from messages import *
from actor.Executor import StreamnetExecutor
from utils.StreamletTools import StreamletTools

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

	def _forward_pass(self, streamlet):
		self.logger.debug("ForwardPass: {0}".format(self.ID))
		streamlets 	  = StreamletTools.shard(streamlet = streamlet, shards = len(self.routees))
		for (each_streamlet, each_routee) in zip(streamlets, self.routees):
			each_streamlet.set_route_to(route_to = self.routes.get_forward_routee())
			each_routee.tell(each_streamlet)

	def _back_prop(self, streamlet):
		self.logger.debug("Backprop: {0}".format(self.ID))
		streamlets 	  = StreamletTools.shard(streamlet = streamlet, shards = len(self.routees))
		for (each_tensor, each_routee) in zip(streamlets, self.routees):
			each_streamlet.set_route_to(route_to = self.routes.get_backprop_routee())
			each_streamlet.set_update_to(route_to = self.routes.get_update_routee())
			each_routee.tell(each_streamlet)

	def _clear_cache(self):
		self.streamlet_cache  = []

	def _handle_forward(self, fs):
		self.streamlet_cache.append(fs)
		if len(self.streamlet_cache) == fs.get_frag():
			streamlet = StreamletTools.merge(streamlets = self.streamlet_cache)
			self._forward_pass(streamlet = streamlet)
			self._clear_cache()

	def _handle_backprop(self, bs):
		self.streamlet_cache.append(bs)
		if len(self.streamlet_cache) == bs.get_frag():
			streamlet = StreamletTools.merge(streamlets = self.streamlet_cache)
			self._back_prop(streamlet = streamlet)
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

