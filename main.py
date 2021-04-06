from actor.DataParallelCoordinator import DataParallelCoordinator
from actor.Dummy import DummyActor
from actor.Activation import StreamnetActivation
from actor.Executor import StreamnetExecutor
from actor.Source import StreamnetSource
from actor.Sink import StreamnetSink
from models.Convolution import Convolution2D
from models.Dense import Dense
from models.Loss import Loss
from models.Sigmoid import Sigmoid
from messages.StartSourceStreamlet import StartSourceStreamlet
import copy
import logging
import tensorflow as tf
import time

def build_dense(units, activation, num_routees):
	executor = StreamnetExecutor.start(deployed_model = Dense(units = units, activation = activation, use_bias = True))
	return DataParallelCoordinator.start(routees = [copy.copy(executor) for _ in range(num_routees)])	

def build_conv(filters, kernel, strides, activation, num_routees):
	executor = StreamnetExecutor.start(deployed_model = Convolution2D(	filters = filters, kernel = kernel, strides = strides, 
																		activation = activation, use_bias = True))
	return DataParallelCoordinator.start(routees = [copy.copy(executor) for _ in range(num_routees)])

def set_proxy_routing(source, layers, sink):
	for indx, each_coordinator in enumerate(layers):	
		proxy = each_coordinator.proxy()
		if len(layers) == 1:
			proxy.routes.set_forward_routee(routee = sink)
			proxy.routes.set_backprop_routee(routee = source)
		elif indx == 0:
			proxy.routes.set_forward_routee(routee = layers[indx + 1])
			proxy.routes.set_backprop_routee(routee = source)
		elif indx == len(layers) -1:		
			proxy.routes.set_forward_routee(routee = sink)
			proxy.routes.set_backprop_routee(routee = layers[indx - 1])
		else:
			proxy.routes.set_forward_routee(routee = layers[indx + 1])
			proxy.routes.set_backprop_routee(routee = layers[indx - 1])
		proxy.routes.set_update_routee(routee = DummyActor.start())
	return

if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO)

	BATCH_SIZE 	= 2**5
	NUM_ROUTEES = 2**2

	source 	= StreamnetSource.start(batch_size = BATCH_SIZE, input_shape = [12,12,3], output_shape = [10,10,8])
	layers 	= [
		build_conv(filters = 10, kernel = (3,3), strides = (1,1), activation = 'sigmoid', num_routees = 1),
		build_dense(units = 8, activation = 'sigmoid', num_routees = NUM_ROUTEES)
	]
	
	loss_model 	= Loss(tf_loss = tf.keras.losses.MeanSquaredError(reduction = tf.keras.losses.Reduction.SUM))
	sink 		= StreamnetSink.start(backward_routee = layers[-1], loss_model = loss_model)
	set_proxy_routing(source = source, layers = layers, sink = sink)
	source.tell(StartSourceStreamlet(route_to = layers[0], sink_route = sink))