import gym
import numpy as np
from gym import spaces
import cv2
from collections import deque


class Environment(object):
	def __init__(self, env_name, image_based=True):

		# TODO implement functionality to self detect whether environment is image_based or not 
		# so user can just pass the environment name and start using the env maybe

		# idea: maybe have another method that first checks if env is image based which then calls this class to 
		# create respective environment

		self.env = gym.make(env_name)

		self.env.seed(0)
		self.action_space = self.env.action_space

		if image_based:
			self.width = 84
			self.height = 84

			# TODO: try using 3, 4, 5, 6 frames to see if results vary
			self.k = 4
			self.observation_space = spaces.Box(low=0, high=255, shape=(self.height, self.width, self.k))
			self.frames = deque([], maxlen=self.k)

	def reset(self):

		obs = self.env.reset()

		if image_based:
			frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
			frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
			obs = frame[:, :, None]
			obs = np.array(obs).astype(np.float32) / 255.0

			for _ in range(self.k):
				self.frames.append(obs)
			obs = np.concatenate(list(self.frames), axis=2)
		return obs

	def step(self,action):
		
		obs, reward, done, info = self.env.step(action)

		if image_based:
			frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
			frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
			obs = frame[:, :, None]
			obs = np.array(obs).astype(np.float32) / 255.0
			self.frames.append(obs)
			if done:
				reward = -5
			return np.concatenate(list(self.frames), axis=2), reward, done, info
		return obs, reward, done, info

	def render(self):
		self.env.render()
