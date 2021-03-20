import pykka
import logging
import messages

class StreamnetExecutor(pykka.ThreadingActor):

	logger 		= logging.getLogger()
	ID 			= None
	model 		= None
	next_routee	= None

	def __init__(self, deployed_model, forward_route, identifier: str = None):
		super().__init__()
		self.ID 			= identifier
		self.model 			= deployed_model
		self.next_routee 	= forward_route
		return

	def on_start(self):
		logging("Starting up Streamnet Executor")

	def on_stop(self):
		logging("Shutting down Streamnet Executor")

	def on_failure(self, exception_type, exception_value, traceback):
		pass

	def on_receive(self, message):
		if message is ForwardStreamlet:
			pass
		elif message is BackpropStreamlet:
			pass

