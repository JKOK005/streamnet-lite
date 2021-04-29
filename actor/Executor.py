import pykka
import logging
import messages
from messages import *

class StreamnetExecutor(pykka.ThreadingActor):

	logger 			= logging.getLogger()

	def __init__(self, 	deployed_model, 
						identifier: str = None):

		super().__init__()
		self.ID 		= identifier
		self.model 		= deployed_model
		self.cur_input 	= None
		return

	def on_start(self):
		self.logger.info("Starting up Streamnet Executor")

	def on_stop(self):
		self.logger.info("Shutting down Streamnet Executor")

	def on_failure(self, exception_type, exception_value, traceback):
		pass

	def on_receive(self, message):
		# Pass the streamlet on to downstream
		if type(message) is messages.ForwardStreamlet:
			tensor 			= message.get_tensor()
			self.logger.debug("Received tensor of shape: {0}".format(tensor.shape))	
			fwd_tensor 		= self.model.forward_pass(tensor = tensor)
			fwd_stream 		= ForwardStreamlet(tensor = fwd_tensor, fragments = message.get_frag(), index = message.get_index())
			self.cur_input 	= tensor

			forward_routee 	= message.get_route_to()
			forward_routee.tell(fwd_stream)

		# Weight / bias updates & send back prop streamlet upstream 
		elif type(message) is messages.BackpropStreamlet:
			tensor 			= message.get_tensor()
			self.logger.debug("Received tensor of shape: {0}".format(tensor.shape))
			bp_tensor 		= self.model.backprop_pass(input_tensor = self.cur_input, dl_dy = tensor)
			wupdt_tensor	= self.model.delta_weights(input_tensor = self.cur_input, dl_dy = tensor)
			bupdt_tensor	= self.model.delta_bias(input_tensor = self.cur_input, dl_dy = tensor)

			bp_stream 		= BackpropStreamlet(tensor = bp_tensor, fragments = message.get_frag(), index = message.get_index())
			wupdt_stream 	= WeightStreamlet(tensor = wupdt_tensor, fragments = message.get_frag(), index = message.get_index())
			bupdt_stream 	= BiasStreamlet(tensor = bupdt_tensor, fragments = message.get_frag(), index = message.get_index())
			bp_routee 		= message.get_route_to()
			update_routee 	= message.get_update_to()

			bp_routee.tell(bp_stream)
			update_routee.tell(wupdt_stream)
			update_routee.tell(bupdt_stream)

		elif type(message) is messages.WeightBiasStreamlet:
			weight_tensor 	= message.weight_streamlet.get_tensor()
			bias_tensor 	= message.bias_streamlet.get_tensor()
			self.model.set_params(weights = weight_tensor, bias = bias_tensor)