import uuid

class Streamlet(object):
	tensor 		= None
	index 		= None
	ID 			= uuid.uuid1() 		# Streamlet unique ID
	fragments 	= 1					# How many streamlets do we expect from the previous layer before the streamlet is complete

	def __init__(self, tensor, fragments, index):
		self.tensor 	= tensor
		self.fragments 	= fragments
		self.index 		= index

	def get_tensor(self):
		return self.tensor

	def get_frag(self):
		return self.fragments

	def get_index(self):
		return self.index