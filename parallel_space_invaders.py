import numpy as np
import gym

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, Dropout, MaxPool2D, Lambda, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
import time
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
import ray

# replace the ray cluster address after starting the cluster head
ray.init(redis_address="192.168.200.150:37460")



# def preprocess_oberservation(img):
# 	img = tf.image.rgb_to_grayscale(img)
# 	img = tf.image.resize(img, (110, 84))
# 	img = tf.image.random_crop(img, (84, 84, 1))

# 	return img

# def extract_batch(imgs):
# 	#imgs = tf.Print(imgs, [tf.shape(imgs)], summarize=10)
# 	imgs = tf.map_fn(preprocess_oberservation, tf.cast(imgs, tf.float32))
# 	return imgs

# def space_invaders_model(dummy_env):
# 	inp = Input(shape=dummy_env.observation_space.shape)

# 	preprocess = Lambda(extract_batch)(inp)
# 	action = Sequential([
#         Conv2D(32, 8, 4, activation='elu'),
#         Conv2D(64, 4, 2, activation='elu'),
#         Conv2D(64, 3, 2, activation='elu'),
#         Flatten(),
#         Dense(512, activation='elu')
# 	])(preprocess)

# 	model = Model(inputs=[inp], outputs=[action])

# 	model.summary()
# 	return model 


def create_keras_model(in_shape, nb_actions):
    tf.compat.v1.disable_eager_execution()

    # Next, we build a very simple model.
    model = Sequential()
    # model.add(Conv2D(32, (8,4)), input_shape = [32,32,in_shape])
    # model.add(Activation('elu'))

    model.add(Flatten(input_shape=in_shape))
    model.add(Dense(16))
    model.add(Activation('elu'))
    model.add(Flatten())
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    print(model.summary())

    return model

# def create_agent():
#     memory = SequentialMemory(limit=50000, window_length=1)
#     policy = BoltzmannQPolicy()
#     dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10, target_model_update=1e-2, policy=policy)
#     dqn.compile(Adam(lr=1e-3), metrics=['mae'])
#     return dqn


@ray.remote
class Network(object):
    def __init__(self):
        tf.compat.v1.disable_eager_execution()

        self.env = gym.make('SpaceInvaders-v0')
        np.random.seed(123)
        self.env.seed(123)
        self.nb_actions = self.env.action_space.n
        #self.model = create_keras_model(((1,) + self.env.observation_space.shape), self.nb_actions)
        self.model = space_invaders_model(self.env)
        self.memory = SequentialMemory(limit=50000, window_length=1)
        self.policy = BoltzmannQPolicy()
        self.dqn = DQNAgent(model=self.model, nb_actions=self.nb_actions, memory=self.memory, nb_steps_warmup=10, target_model_update=1e-2, policy=self.policy)
        self.dqn.compile(Adam(lr=1e-3), metrics=['mae'])


# Method done here : reduce the number of steps and run multiple workers and share the weights between these workers

    def train(self):
        # Here it is training every process for number of steps, need to divide the the 50000 steps on different processes
        history = self.dqn.fit(self.env, nb_steps=10000, visualize=True, verbose=2)
        return history.history

    def get_weights(self):
        # "weights_{}_{}_{}.h5".format(START_TIME, e, avg_reward)
        self.dqn.save_weights('dqn_{}_weights.h5'.format('SpaceInvaders-v0'), overwrite=True)
        return self.model.get_weights()

    def set_weights(self, weights):
        # Note that for simplicity this does not handle the optimizer state.
        self.model.set_weights(weights)

# Initiating the worker nodes and exchanging the weights between the workers

actor1 = Network.remote()
actor1.train.remote()
# result_object_id = actor1.train.remote()
# ray.get(result_object_id)


actor2 = Network.remote()
actor2.train.remote()
weights = ray.get(
    [actor1.get_weights.remote(),
    actor2.get_weights.remote()])

averaged_weights = [(layer1 + layer2)/2 for layer1, layer2 in zip(weights[0], weights[1])]

weight_id = ray.put(averaged_weights)
[
    actor.set_weights.remote(weight_id) for actor in [actor1, actor2]
]

ray.get([actor.train.remote() for actor in [actor1, actor2]])

# Observatios:
# 1. Loss is high
# 2. No epochs 
