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
import tensorflow as tf
import time

if __name__ == "__main__":
	import logging
	logging.basicConfig(level=logging.INFO)

	source 				= StreamnetSource.start()

	executor 			= StreamnetExecutor.start(deployed_model = Dense(units = 64, activation = None, use_bias = True))
	dp_coordinator_1 	= DataParallelCoordinator.start(routees = [copy.copy(executor) for _ in range(2**7)])	

	activation 			= StreamnetActivation.start(activation = Sigmoid())
	dp_coordinator_2 	= DataParallelCoordinator.start(routees = [copy.copy(activation)])

	executor 			= StreamnetExecutor.start(deployed_model = Dense(units = 32, activation = None, use_bias = True))
	dp_coordinator_3 	= DataParallelCoordinator.start(routees = [copy.copy(executor) for _ in range(2**7)])

	activation 			= StreamnetActivation.start(activation = Sigmoid())
	dp_coordinator_4 	= DataParallelCoordinator.start(routees = [copy.copy(activation)])

	executor			= StreamnetExecutor.start(deployed_model = Dense(units = 16, activation = None, use_bias = True))
	dp_coordinator_5 	= DataParallelCoordinator.start([copy.copy(executor) for _ in range(2**7)])

	activation 			= StreamnetActivation.start(activation = Sigmoid())
	dp_coordinator_6 	= DataParallelCoordinator.start(routees = [copy.copy(activation)])

	loss_model 			= Loss(tf_loss = tf.keras.losses.MeanSquaredError(reduction = tf.keras.losses.Reduction.SUM))
	sink 				= StreamnetSink.start(backward_routee = dp_coordinator_6, loss_model = loss_model)

	proxy = dp_coordinator_1.proxy()
	proxy.routes.set_forward_routee(routee = dp_coordinator_2)
	proxy.routes.set_backprop_routee(routee = source)
	proxy.routes.set_update_routee(routee = DummyActor.start())

	proxy = dp_coordinator_2.proxy()
	proxy.routes.set_forward_routee(routee = dp_coordinator_3)
	proxy.routes.set_backprop_routee(routee = dp_coordinator_1)

	proxy = dp_coordinator_3.proxy()
	proxy.routes.set_forward_routee(routee = dp_coordinator_4)
	proxy.routes.set_backprop_routee(routee = dp_coordinator_2)
	proxy.routes.set_update_routee(routee = DummyActor.start())

	proxy = dp_coordinator_4.proxy()
	proxy.routes.set_forward_routee(routee = dp_coordinator_5)
	proxy.routes.set_backprop_routee(routee = dp_coordinator_3)

	proxy = dp_coordinator_5.proxy()
	proxy.routes.set_forward_routee(routee = dp_coordinator_6)
	proxy.routes.set_backprop_routee(routee = dp_coordinator_4)
	proxy.routes.set_update_routee(routee = DummyActor.start())

	proxy = dp_coordinator_6.proxy()
	proxy.routes.set_forward_routee(routee = sink)
	proxy.routes.set_backprop_routee(routee = dp_coordinator_5)

	time.sleep(1)
	source.tell(StartSourceStreamlet(route_to = dp_coordinator_1, sink_route = sink))