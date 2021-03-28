from messages.Streamlet import Streamlet

class StartSourceStreamlet():
	route_to 	= None 		# This field may be None
	interval 	= 0

	def __init__(self, route_to = None, interval = 1):
		self.route_to = route_to
		self.interval = interval

	# Address of upstream layer
	def get_route_to(self):
		return self.route_to

	def get_interval(self):
		return self.interval