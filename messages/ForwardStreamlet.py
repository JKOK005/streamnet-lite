from messages.Streamlet import Streamlet

class ForwardStreamlet(Streamlet):
	route_to 	= None 		# This field may be None

	def __init__(self, tensor, fragments, index, route_to = None):
		super(ForwardStreamlet, self).__init__(tensor = tensor, fragments = fragments, index = index)
		self.route_to = route_to

	# Address of upstream layer
	def get_route_to(self):
		return self.route_to

	def set_route_to(self, route_to):
		self.route_to = route_to