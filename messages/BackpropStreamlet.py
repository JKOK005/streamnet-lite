from messages.Streamlet import Streamlet

class BackpropStreamlet(Streamlet):
	def __init__(self, tensor):
		super(BackpropStreamlet, self).__init__(tensor = tensor)