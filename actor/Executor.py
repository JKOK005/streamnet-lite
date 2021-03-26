import pykka
import logging
from messages import *

class StreamnetExecutor(pykka.ThreadingActor):

	logger 			= logging.getLogger()
	ID 				= None
	model 			= None
	cur_input 		= None

	def __init__(self, 	deployed_model, 
						identifier: str = None):

		super().__init__()
		self.ID 				= identifier
		self.model 				= deployed_model
		return

	def on_start(self):
		logging("Starting up Streamnet Executor")

	def on_stop(self):
		logging("Shutting down Streamnet Executor")

	def on_failure(self, exception_type, exception_value, traceback):
		pass

	def on_receive(self, message):
		tensor = message.get_tensor()
		logger.info("Received tensor of shape: {0}".format(tensor.shape))

		# Pass the streamlet on to downstream
		if message is ForwardStreamlet:
			fwd_tensor 		= deployed_model.forward_pass(tensor = tensor)
			fwd_stream 		= ForwardStreamlet(tensor = fwd_tensor)
			self.cur_input 	= tensor

			forward_routee 	= message.get_route_to()
			forward_routee.tell(fwd_stream)

		# Weight / bias updates & send back prop streamlet upstream 
		elif message is BackpropStreamlet:
			bp_tensor 		= deployed_model.backprop_pass(upstream_backprop_tensor = tensor)
			wupdt_tensor	= deployed_model.delta_weights(input_tensor = cls.cur_input, upstream_backprop_tensor = tensor)
			bupdt_tensor	= deployed_model.delta_bias(upstream_backprop_tensor = tensor)

			bp_stream 		= BackpropStreamlet(tensor = bp_tensor)
			bp_routee 		= message.get_route_to()

			wupdt_stream 	= WeightStreamlet(tensor = wupdt_tensor)
			bupdt_stream 	= BiasStreamlet(tensor = bupdt_tensor)
			update_routee 	= message.get_update_to()

			bp_routee.tell(bp_stream)
			update_routee.tell(wupdt_stream)
			update_routee.tell(bupdt_stream)