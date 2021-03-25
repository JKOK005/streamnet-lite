from messages.Streamlet import Streamlet

class WeightStreamlet(Streamlet):
	def __init__(self, tensor, fragments):
		super(WeightStreamlet, self).__init__(tensor = tensor, fragments = fragments)