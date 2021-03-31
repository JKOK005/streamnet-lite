from messages.Streamlet import Streamlet

class LabelStreamlet(Streamlet):

	def __init__(self, tensor, fragments, index):
		super(LabelStreamlet, self).__init__(tensor = tensor, fragments = fragments, index = index)
