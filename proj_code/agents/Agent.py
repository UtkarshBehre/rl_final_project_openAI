import os
import numpy as np
import gym
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Activation, Flatten
from keras.optimizers import Adam
from collections import deque
import random
import os
import time

class Agent:

	def __init__(self):
		self.hasEpsilon = False


	
	def test_agent(self, total_runs):
        '''
        tests the agent by running it total_runs number of times against the model
        '''
        if self.hasEpsilon:
        	self.epsilon = self.epsilon_min

        best_score = 0
        for run in range(1, total_runs+1):
            run_score = 0
            done = False
            state = self.env.reset()
            
            while not done:
                next_state, reward, done, _ = self.env.step(self.take_action(state))
                run_score += reward
                state = next_state
            
            print("Agent: %s \t Environment: %s \t Run: %i/%i | Score: %i" % (self.agent_name, self.env_name, run, total_runs, run_score), end="\r")
            best_score = best_score if best_score>run_score else run_score

        print("%s Agent's High Score on %s Environment in %i runs: %i " %(self.agent_name, self.env_name, total_runs, best_score))