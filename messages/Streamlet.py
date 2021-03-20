import uuid

class Streamnet(object):
	tensor 	= None
	ID 		= uuid.uuid1() 		# Streamlet unique ID

	def __init__(self, tensor):
		self.tensor = tensor

	def get_tensor(self):
		return tensor