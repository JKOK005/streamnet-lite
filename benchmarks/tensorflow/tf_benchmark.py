import tensorflow as tf 
from tensorflow.keras.layers import Input, Conv2D, Dense
from tensorflow.keras import Model

if __name__ == "__main__":
	tf.config.threading.set_intra_op_parallelism_threads(1)
	tf.config.threading.set_inter_op_parallelism_threads(1)

	inputs 	= Input(shape = (24, 24, 3), name = "input", dtype = tf.float32)
	conv_1 	= Conv2D(filters = 32, kernel_size = (3,3), strides = (1,1), activation = "sigmoid")(inputs)
	conv_2 	= Conv2D(filters = 32, kernel_size = (3,3), strides = (1,1), activation = "sigmoid")(conv_1)
	conv_3 	= Conv2D(filters = 32, kernel_size = (3,20), strides = (1,1), activation = "sigmoid")(conv_2)
	dense_1 = Dense(units = 16, activation='sigmoid', use_bias=True)(conv_3)
	dense_2 = Dense(units = 16, activation='sigmoid', use_bias=True)(dense_1)
	dense_3 = Dense(units = 16, activation='sigmoid', use_bias=True)(dense_2)
	model 	= Model(inputs = [inputs], outputs = [dense_3])
	model.compile(optimizer='adam', loss=tf.losses.MeanSquaredError(), metrics=['accuracy'])

	print(model.summary())

	x = tf.random.uniform([2**10, 24, 24, 3])
	y = tf.random.uniform([2**10, 18, 1, 16])

	history = model.fit(
	    x = x, 
	    y = y,
	    batch_size=32,
	    validation_split=0.0,
	    validation_data=None, 
	    epochs=100, 
	    max_queue_size=5,
	    workers=2,
	)