from messages.Streamlet import Streamlet

class BiasStreamlet(Streamlet):
	def __init__(self, tensor, fragments):
		super(BiasStreamlet, self).__init__(tensor = tensor, fragments = fragments)