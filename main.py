from actor.DataParallelCoordinator import DataParallelCoordinator
from actor.Executor import StreamnetExecutor
from actor.Source import StreamnetSource
from actor.Sink import StreamnetSink
from actor.Dummy import DummyActor
from models.Dense import Dense
from models.Loss import Loss
from messages.StartSourceStreamlet import StartSourceStreamlet
import copy
import tensorflow as tf
import time

if __name__ == "__main__":
	import logging
	logging.basicConfig(level=logging.INFO)

	source 				= StreamnetSource.start()

	model_1 			= Dense(units = 64, activation = None, use_bias = True)
	executor_1 			= StreamnetExecutor.start(deployed_model = model_1)
	dp_coordinator_1 	= DataParallelCoordinator.start(routees = [copy.copy(executor_1) for _ in range(2**7)])	

	model_2 			= Dense(units = 32, activation = None, use_bias = True)
	executor_2 			= StreamnetExecutor.start(deployed_model = model_2)
	dp_coordinator_2 	= DataParallelCoordinator.start(routees = [copy.copy(executor_2) for _ in range(2**7)])

	model_3 			= Dense(units = 16, activation = None, use_bias = True)
	executor_3			= StreamnetExecutor.start(deployed_model = model_3)
	dp_coordinator_3 	= DataParallelCoordinator.start([copy.copy(executor_3) for _ in range(2**7)])

	loss_model 			= Loss(tf_loss = tf.keras.losses.MeanSquaredError(reduction = tf.keras.losses.Reduction.SUM))
	sink 				= StreamnetSink.start(backward_routee = dp_coordinator_3, loss_model = loss_model)

	proxy_1 = dp_coordinator_1.proxy()
	proxy_1.routes.set_forward_routee(routee = dp_coordinator_2)
	proxy_1.routes.set_backprop_routee(routee = source)
	proxy_1.routes.set_update_routee(routee = DummyActor.start())

	proxy_2 = dp_coordinator_2.proxy()
	proxy_2.routes.set_forward_routee(routee = dp_coordinator_3)
	proxy_2.routes.set_backprop_routee(routee = dp_coordinator_1)
	proxy_2.routes.set_update_routee(routee = DummyActor.start())

	proxy_3 = dp_coordinator_3.proxy()
	proxy_3.routes.set_forward_routee(routee = sink)
	proxy_3.routes.set_backprop_routee(routee = dp_coordinator_2)
	proxy_3.routes.set_update_routee(routee = DummyActor.start())

	time.sleep(1)
	source.tell(StartSourceStreamlet(route_to = dp_coordinator_1, sink_route = sink))