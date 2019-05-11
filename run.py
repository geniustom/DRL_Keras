# -*- coding: utf-8 -*-
import gym
import tensorflow as tf
from A3C.a3c import A3C

from utils.atari_environment import AtariEnvironment
from utils.continuous_environments import Environment
from utils.networks import get_session
from keras.backend.tensorflow_backend import set_session

class Arg():
	def __init__(self):
		self.env='CartPole-v1'
		self.consecutive_frames=4
		self.nb_episodes=10000
		self.training_interval=30
		self.n_threads=16
		self.is_atari=False
		self.render=False
	


def main():
	args=Arg()
	env = Environment(gym.make(args.env), args.consecutive_frames)
	env.reset()
	state_dim = env.get_state_size()
	action_dim = gym.make(args.env).action_space.n
	algo = A3C(action_dim, state_dim, args.consecutive_frames, is_atari=args.is_atari)
	set_session(get_session())
	summary_writer = tf.summary.FileWriter("./tensorboard_" + args.env)
	stats = algo.train(env, args, summary_writer)
	
	
	print(stats)
	algo.save_weights('./'+args.env+'.h5')
	env.env.close()
	
	
	
if __name__ == "__main__":
    main()