import uuid

class Streamlet(object):
	tensor 		= None
	ID 			= uuid.uuid1() 		# Streamlet unique ID
	fragments 	= 1					# How many streamlets do we expect from the previous layer before the streamlet is complete

	def __init__(self, tensor, fragments):
		self.tensor 	= tensor
		self.fragments 	= fragments

	def get_tensor(self):
		return self.tensor

	def get_frag(self):
		return self.fragments