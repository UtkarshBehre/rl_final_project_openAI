import gym
import numpy as np
from gym import spaces
import cv2
from collections import deque


class Environment(object):
	def __init__(self, env_name, image_based):

		self.env = gym.make(env_name)
		self.name = env_name
		self.image_based = image_based

		self.env.seed(0)
		self.action_space = self.env.action_space.n
		self.state_size = self.env.observation_space.shape
		if self.image_based:
			self.width = 84
			self.height = 84

			# TODO: try using 3, 4, 5, 6 frames to see if results vary
			self.k = 4
			self.state_size = spaces.Box(low=0, high=255, shape=(self.height, self.width, self.k))
			self.frames = deque([], maxlen=self.k)

	def reset(self):

		obs = self.env.reset()

		if self.image_based:
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

		if self.image_based:
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
