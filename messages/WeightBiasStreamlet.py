class WeightBiasStreamlet(object):
	def __init__(self, reduced_weight_streamlet, reduced_bias_streamlet):
		self.weight_streamlet 	= reduced_weight_streamlet
		self.bias_streamlet 	= reduced_bias_streamlet