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
	def _handle_forward(self, fs):
		pass

	@abstractmethod
	def _handle_backprop(self, bs):
		pass

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
			pass
		elif message is BackpropStreamlet:
			pass

