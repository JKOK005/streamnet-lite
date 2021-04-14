from numpy import array
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import tensorflow_datasets as tfds
import tensorflow as tf

class Cifar10DSGen(object):
	def __init__(self, num_samples: int = None):
		 self.CIFAR10_DS = tfds.load("cifar10", split = "train", shuffle_files = False)
		 if num_samples is not None:
		 	self.CIFAR10_DS = self.CIFAR10_DS.take(num_samples)

		 class_categories 	= array([i for i in range(10)])
		 label_encoder 		= LabelEncoder()
		 integer_encoded 	= label_encoder.fit_transform(class_categories)
		 onehot_encoder 	= OneHotEncoder(sparse=False)
		 integer_encoded 	= integer_encoded.reshape(len(integer_encoded), 1)
		 onehot_encoded 	= onehot_encoder.fit_transform(integer_encoded)
		 self.cat_dict 		= { k : v for (k , v) in zip(class_categories, onehot_encoded) }
		 return

	def gen_ds(self, batch_size: int):
		while True:
			for each_batch in self.CIFAR10_DS.batch(batch_size):
				imgs 	= each_batch["image"]
				labels 	= each_batch["label"]
				onehot_labels = tf.convert_to_tensor([self.cat_dict[i] for i in labels.numpy()])
				yield(imgs, onehot_labels)