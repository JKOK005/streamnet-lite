from messages.Streamlet import Streamlet

class WeightStreamlet(Streamlet):
	def __init__(self, tensor, fragments, index):
		super(WeightStreamlet, self).__init__(tensor = tensor, fragments = fragments, index = index)