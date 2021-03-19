import pykka
import logging

class StreamnetCoordinator(pykka.ThreadingActor):

	logger 	= logging.getLogger()
	routees = None

	def __init__(self, routees):
		super().__init__()
		self.routees = routees
		pass

	def on_start(self):
		logging("Starting up Streamnet Executor")

	def on_stop(self):
		logging("Shutting down Streamnet Executor")

	def on_failure(self, exception_type, exception_value, traceback):
		pass

	def on_receive(self, message):
		pass

