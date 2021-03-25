from messages.Streamlet import Streamlet

class ForwardStreamlet(Streamlet):
	def __init__(self, tensor, fragments):
		super(ForwardStreamlet, self).__init__(tensor = tensor, fragments = fragments)