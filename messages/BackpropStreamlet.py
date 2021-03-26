from messages.Streamlet import Streamlet

class BackpropStreamlet(Streamlet):
	route_to 	= None		# This field may be None
	update_to 	= None		# This field may be None

	def __init__(self, tensor, fragments, route_to = None, update_to = None):
		super(BackpropStreamlet, self).__init__(tensor = tensor, fragments = fragments)
		self.route_to 	= route_to
		self.update_to 	= update_to

	# Address of upstream layer
	def get_route_to(self):
		return self.route_to

	# Address of optimizer
	def get_update_to(self):
		return self.update_to