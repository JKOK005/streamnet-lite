import pykka
import logging
import uuid
import messages
from abc import ABCMeta
from actor import Executor

class StreamnetCoordinator(pykka.ThreadingActor, metaclass = ABCMeta):

	logger 	= logging.getLogger()
	routees = None
	ID 		= None
	streamlet_cache 	= []

	def __init__(self, routees: [Executor], identifier: str = None):
		super().__init__()
		self.routees 	= routees
		self.ID 		= identifier if identifier else str(uuid.uuid1())
		return

	@abstractmethod
	def _scale_up(self):
		pass

	@abstractmethod
	def _scale_down(self):
		pass

	@abstractmethod
	def _forward_pass(self, tensor):
		pass

	@abstractmethod
	def _back_prop(self, tensor):
		pass

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
		logging("Starting up Streamnet Coordinator: {0}".format(self.ID))

	def on_stop(self):
		logging("Shutting down Streamnet Coordinator: {0}".format(self.ID))

	def on_failure(self, exception_type, exception_value, traceback):
		pass

	def on_receive(self, message):
		if message is ScaleRoutee:
			pass
		
		elif message is ShrinkRoutee:
			pass

		elif message is ForwardStreamlet:
			self._handle_forward(fs = message)

		elif message is BackpropStreamlet:
			self._handle_backprop(bs = message)

