from actor.DataParallelCoordinator import DataParallelCoordinator
from actor.Executor import StreamnetExecutor
from actor.Source import StreamnetSource
from actor.Sink import StreamnetSink
from actor.Dummy import DummyActor
from models.Dense import Dense
import copy
import tensorflow as tf

if __name__ == "__main__":
	dp_coordinator_1 	= DataParallelCoordinator()
	dp_coordinator_2 	= DataParallelCoordinator()
	dp_coordinator_3 	= DataParallelCoordinator()

	dummy_1 			= DummyActor()
	dummy_2 			= DummyActor()
	dummy_3 			= DummyActor()

	source 				= StreamnetSource(forward_routee = dp_coordinator_1)
	sink 				= StreamnetSink(backward_routee = dp_coordinator_3)

	model_1 			= tf.keras.layers.Dense(units = 7, activation = "relu", use_bias = True)
	executor_1 			= StreamnetExecutor(
							deployed_model 	= model_1,
							forward_routee 	= dp_coordinator_2,
							backward_routee = source,
							update_routee  	= dummy_1,
						)
	dp_coordinator_1.add_routees([copy.copy(executor_1) for _ in range(1)])

	model_2 			= tf.keras.layers.Dense(units = 5, activation = "relu", use_bias = True)
	executor_2 			= StreamnetExecutor(
							deployed_model 	= model_2,
							forward_routee 	= dp_coordinator_3,
							backward_routee = dp_coordinator_2,
							update_routee  	= dummy_2,
						)
	dp_coordinator_2.add_routees([copy.copy(executor_2) for _ in range(1)])

	model_3 			= tf.keras.layers.Dense(units = 3, activation = "relu", use_bias = True)
	executor_3			= StreamnetExecutor(
							deployed_model 	= model_3,
							forward_routee 	= sink,
							backward_routee = dp_coordinator_3,
							update_routee  	= dummy_3,
						)
	dp_coordinator_3.add_routees([copy.copy(executor_3) for _ in range(1)])

	source.initiate(interval = 30)