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
		tensors 		= [each_streamlet.get_tensor() for each_streamlet in streamlets]
		indexes 		= [each_streamlet.get_index() for each_streamlet in streamlets]
		full_tensor 	= tf.concat(tensors, axis = 0)
		full_index 		= tf.concat(indexes, axis = 0)
		
		# Merge automatically sorts by index
		sorted_pair		= sorted(zip(full_tensor, full_index), key = lambda row: row[-1], reverse = False)
		ordered_tensor  = tf.convert_to_tensor(list(map(lambda row: row[0], sorted_pair)))
		sort_index 		= tf.convert_to_tensor([i for i in range(len(ordered_tensor))])
		return type(streamlets[0])(tensor = ordered_tensor, fragments = 1, index = sort_index)

	@staticmethod
	def reduce_on_batch(streamlets):
		tensors 	= [each_streamlet.get_tensor() for each_streamlet in streamlets]
		full_tensor = tf.concat(tensors, axis = 0)
		full_tensor = tf.math.reduce_mean(full_tensor, axis = 0)
		return type(streamlets[0])(tensor = full_tensor, fragments = 1, index = None)

	@staticmethod
	def join_on_index(streamlet_A, streamlet_B):
		tensor_A 	 	 = streamlet_A.get_tensor()
		index_A 		 = streamlet_A.get_index()
		sorted_pair_A 	 = sorted(zip(tensor_A, index_A), key = lambda row: row[-1], reverse = False)
		ordered_tensor_A = tf.convert_to_tensor(list(map(lambda row: row[0], sorted_pair_A)))

		tensor_B 		 = streamlet_B.get_tensor()
		index_B 		 = streamlet_B.get_index()
		sorted_pair_B 	 = sorted(zip(tensor_B, index_B), key = lambda row: row[-1], reverse = False)
		ordered_tensor_B = tf.convert_to_tensor(list(map(lambda row: row[0], sorted_pair_B)))

		sort_index 		 = tf.convert_to_tensor([i for i in range(len(ordered_tensor_A))])
		return (ordered_tensor_A, ordered_tensor_B, sort_index)