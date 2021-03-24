import messages
import tensorflow as tf
from actor.Coordinator import Coordinator

class DataParallelCoordinator(Coordinator):
	
	def __init__(self, routees: [Executor], identifier: str = None):
		super(DataParallelCoordinator).__init__(routees = routees, identifier = identifier)

	def _scale_up(self):
		pass

	def _scale_down(self):
		pass

	"""
	Splits the data by the batch dimension 
	Splits are determined by the number of routees
	"""
	def _forward_pass(self, tensor):
		split_tensors = tf.split(tensor, len(self.routees), axis = 0)
		for (each_tensor, each_routee) in zip(split_tensors, self.routees):
			each_routee.tell(ForwardStreamlet(tensor = each_tensor))

	def _back_prop(self, tensor):
		for each_routee in self.routees:
			each_routee.tell(BackpropStreamlet(tensor = tensor))