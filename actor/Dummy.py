import pykka
import logging
from messages import *
import time

class DummyActor(pykka.ThreadingActor):
	logger 			= logging.getLogger()

	def on_start(self):
		logging("Starting up Dummy actor")

	def on_stop(self):
		logging("Shutting down Dummy actor")

	def on_receive(self, message):		
		self.logger.info("Dummy actor received streamlet")