from actor.DataParallelCoordinator import DataParallelCoordinator
from actor.Dummy import DummyActor
from actor.Activation import StreamnetActivation
from actor.Executor import StreamnetExecutor
from actor.Source import StreamnetSource
from actor.Sink import StreamnetSink
from models.Dense import Dense
from models.Loss import Loss
from models.Sigmoid import Sigmoid
from messages.StartSourceStreamlet import StartSourceStreamlet
import copy
import logging
import tensorflow as tf
import time

def build_dense(units, num_routees):
	executor = StreamnetExecutor.start(deployed_model = Dense(units = units, use_bias = True))
	return DataParallelCoordinator.start(routees = [copy.copy(executor) for _ in range(num_routees)])	

def build_activation(activation, num_routees):
	activation = StreamnetActivation.start(activation = activation)
	return DataParallelCoordinator.start(routees = [copy.copy(activation)])

def set_proxy_routing(source, layers, sink):
	for indx, each_coordinator in enumerate(layers):	
		proxy = each_coordinator.proxy()
		if 	indx == 0:
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

	BATCH_SIZE 	= 2**12
	NUM_ROUTEES = 2**0

	source 	= StreamnetSource.start(batch_size = BATCH_SIZE, input_shape = [128,128], output_shape = [128,128])
	layers 	= [
		build_dense(units = 128, num_routees = NUM_ROUTEES),
		build_activation(activation = Sigmoid(), num_routees = NUM_ROUTEES),
	]
	
	loss_model 	= Loss(tf_loss = tf.keras.losses.MeanSquaredError(reduction = tf.keras.losses.Reduction.SUM))
	sink 		= StreamnetSink.start(backward_routee = layers[-1], loss_model = loss_model)
	set_proxy_routing(source = source, layers = layers, sink = sink)
	source.tell(StartSourceStreamlet(route_to = layers[0], sink_route = sink))