import tensorflow as tf

class StreamletTools(object):
	@staticmethod
	def shard(streamlet, shards):
		tensor 			= streamlet.get_tensor()
		index 			= streamlet.get_index()
		tensor_splits 	= tf.split(tensor, shards, axis = 0)
		index_splits 	= tf.split(index, shards, axis = 0)
		return 	[
					type(streamlet)(tensor = each_tensor, fragments = shards, index = each_index)
					for (each_tensor, each_index) in zip(tensor_splits, index_splits)
				]

	@staticmethod
	def merge(streamlets):
		tensors 	= [each_streamlet.get_tensor() for each_streamlet in streamlets]
		indexes 	= [each_streamlet.get_index() for each_streamlet in streamlets]
		full_tensor = tf.concat(tensors, axis = 0)
		full_index 	= tf.concat(indexes, axis = 0)
		return type(streamlets[0])(tensor = full_tensor, fragments = 1, index = full_index)

	@staticmethod
	def join_on_index(streamlet_A, streamlet_B):
		pass