import pykka
import logging
import messages
from messages import *
import time

class DummyActor(pykka.ThreadingActor):
	logger 			= logging.getLogger()

	def on_start(self):
		self.logger.info("Starting up Dummy actor")

	def on_stop(self):
		self.logger.info("Shutting down Dummy actor")

	def on_receive(self, message):		
		self.logger.debug("Dummy actor received streamlet")