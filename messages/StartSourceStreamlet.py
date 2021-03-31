from messages.Streamlet import Streamlet

class StartSourceStreamlet():
	route_to 	= None 		# This field may be None
	sink_route 	= None

	def __init__(self, route_to = None, sink_route = None):
		self.route_to 	= route_to
		self.sink_route = sink_route

	# Address of upstream layer
	def get_route_to(self):
		return self.route_to

	def get_sink_route(self):
		return self.sink_route