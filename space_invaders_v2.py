import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.applications import MobileNetV2
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPool2D, Lambda, Input

import matplotlib.pyplot as plt
import gym
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import time
import numpy as np

print(tf.executing_eagerly())

import huskarl as hk

INSTANCES = 1
BATCH_SIZE = 8
WINDOW_SIZE = 4
EPOCHS = 30
START_TIME = int(time.time())

def preprocess_oberservation(img):
	img = tf.image.rgb_to_grayscale(img)
	img = tf.image.resize(img, (110, 84))
	img = tf.image.random_crop(img, (84, 84, 1))

	return img

def space_invaders_model(dummy_env):
	inp = Input(shape=dummy_env.observation_space.shape)

	preprocess = Lambda(extract_batch)(inp)
	action = Sequential([
        Conv2D(32, 8, 4, activation='elu'),
        Conv2D(64, 4, 2, activation='elu'),
        Conv2D(64, 3, 2, activation='elu'),
        Flatten(),
        Dense(512, activation='elu')
	])(preprocess)

	model = Model(inputs=[inp], outputs=[action])

	model.summary()
	return model 

def extract_batch(imgs):
	#imgs = tf.Print(imgs, [tf.shape(imgs)], summarize=10)
	imgs = tf.map_fn(preprocess_oberservation, tf.cast(imgs, tf.float32))
	return imgs

if __name__ == '__main__':
	# Setup gym environment
	# env = gym.make('SpaceInvaders-v0')
	create_env = lambda: gym.make('SpaceInvaders-v0').unwrapped
	dummy_env = create_env()

	# Build a simple neural network with 3 fully connected layers as our model
	print(dummy_env.observation_space.shape)


	# log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
	NAME = "space_invaders_dqn-{}.".format(int(time.time()))
	# Create Deep Q-Learning Network agent
	agent = hk.agent.DQN(space_invaders_model(dummy_env), actions=dummy_env.action_space.n, nsteps=WINDOW_SIZE, batch_size=BATCH_SIZE)

	# rl.agents.dqn.DQNAgent(model, policy=None, test_policy=None, enable_double_dqn=True, enable_dueling_network=False, dueling_type='avg')


	avg_reward = 0
	def plot_rewards(episode_rewards, episode_steps, done=False):
		global avg_reward
		print(episode_rewards, episode_steps, done)
		avg_reward = int(np.array(episode_rewards).mean())


	# Create simulation, train and then test
	# sim = hk.Simulation(create_env, agent)
	sim = hk.Simulation(create_env, agent)

	for e in range(EPOCHS):

		sim.train(max_steps=20_000, visualize=True, plot=plot_rewards, max_subprocesses=INSTANCES, instances=INSTANCES)
		agent.save("weights_{}_{}_{}.h5".format(START_TIME, e, avg_reward))


	sim.test(max_steps=1000) 