import pykka
import logging
import messages

class StreamnetExecutor(pykka.ThreadingActor):

	logger 			= logging.getLogger()
	ID 				= None
	model 			= None
	forward_routee	= None
	backward_routee = None
	update_routee 	= None
	cur_input 		= None

	def __init__(self, 	deployed_model, 
						forward_routee, 
						backward_routee, 
						update_routee, 
						identifier: str = None):

		super().__init__()
		self.ID 				= identifier
		self.model 				= deployed_model
		self.forward_routee 	= forward_routee
		self.backward_routee 	= backward_routee
		self.update_routee 		= update_routee
		return

	def on_start(self):
		logging("Starting up Streamnet Executor")

	def on_stop(self):
		logging("Shutting down Streamnet Executor")

	def on_failure(self, exception_type, exception_value, traceback):
		pass

	def on_receive(self, message):
		tensor = message.get_tensor()

		# Pass the streamlet on to downstream
		if message is ForwardStreamlet:
			fwd_tensor 		= deployed_model.forward_pass(tensor = tensor)
			fwd_stream 		= ForwardStreamlet(tensor = fwd_tensor)
			self.cur_input 	= tensor
			self.forward_routee.tell(fwd_stream)

		# Weight / bias updates & send back prop streamlet upstream 
		elif message is BackpropStreamlet:
			bp_tensor 		= deployed_model.backprop_pass(upstream_backprop_tensor = tensor)
			wupdt_tensor	= deployed_model.delta_weights(input_tensor = cls.cur_input, upstream_backprop_tensor = tensor)
			bupdt_tensor	= deployed_model.delta_bias(upstream_backprop_tensor = tensor)

			bp_stream 		= BackpropStreamlet(tensor = bp_tensor)
			wupdt_stream 	= WeightStreamlet(tensor = wupdt_tensor)
			bupdt_stream 	= BiasStreamlet(tensor = bupdt_tensor)

			self.backward_route.tell(bp_stream)
			self.update_routee.tell(wupdt_stream)
			self.update_routee.tell(bupdt_stream)