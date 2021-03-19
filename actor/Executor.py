import pykka
import logging

class StreamnetExecutor(pykka.ThreadingActor):

	logger = logging.getLogger()

	def __init__(self, deployed_model, forward_routees):
		super().__init__()
		pass

	def on_start(self):
		logging("Starting up Streamnet Executor")

	def on_stop(self):
		logging("Shutting down Streamnet Executor")

	def on_failure(self, exception_type, exception_value, traceback):
		pass

	def on_receive(self, message):
		pass

