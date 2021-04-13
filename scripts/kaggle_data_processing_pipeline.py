import cv2
import glob
import tensorflow as tf

CAT_LABELS = tf.convert_to_tensor([1,0])
DOG_LABELS = tf.convert_to_tensor([0,1])

if __name__ == "__main__":
	img_tensors = []
	img_classes = []

	all_images 	= glob.glob("./dataset/kaggle/*.jpg")
	for each_image in all_images:
		img_data 	= cv2.imread(each_image)
		img_tensor 	= tf.convert_to_tensor(img_data)
		img_tensors.append(img_tensor)

		if "cat" in each_image:
			img_classes.append(CAT_LABELS)
		else:
			img_classes.append(DOG_LABELS)

	
	import IPython
	IPython.embed()

	full_img_tensor = tf.stack(img_tensors, axis = 0)
	full_img_label 	= tf.stack(img_classes, axis = 0)
