import gym
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
import random
import os

class DQNAgent_0c:
    
    def __init__(self, env_name, memlen=2000):

        self.env = gym.make(env_name)

        self.state_dim = self.env.observation_space.shape[0]
        self.actions_dim = self.env.action_space.n
        
        self.memory = deque(maxlen=memlen)
        self.batch_size = 32

        self.learning_rate = 0.001
        self.gamma = 0.95
        
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        
        # evaluation and target model initiallizations
        self.model_eval = self.build_model()
        self.model_target = self.build_model()

        self.start_after = 10
        self.update_cycle = 20
        self.weights_path = "cartpole_weights.h5"

    def build_model(self):
        
        model = Sequential()
        
        model.add(Dense(24, input_dim = self.state_dim, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.actions_dim, activation='linear'))
        
        # cost functions = MSE, adam optimizer works well for cartpole environment
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        
        return model
    
    def memorize(self, state, action, reward, next_state, done):
        '''
        stores state transitions for actions
        '''
        # reshape required for keras model
        state = np.reshape(state, [1, self.state_dim])
        next_state = np.reshape(next_state, [1,self.state_dim])

        self.memory.append((state, action, reward, next_state, done))
    
    # figures out which action to take given the state
    # exploration and exploitation
    def take_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(0, self.actions_dim)
        else:
            state = np.reshape(state, [1, self.state_dim]) # adding batch dim for model
            return np.argmax(self.model_eval.predict(state)[0])
    
    # trains the network based on memorized gameplay
    def learn_from_memory(self):
        
        if len(self.memory)<self.batch_size:
            return
        
        # sample from the recorded memories instead of using all the memories to ensure generalization
        minibatch = random.sample(self.memory, self.batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            q_future_by_target = self.model_target.predict(next_state)
            
            # current reward + discounted future reward * estimate of future reward
            q_to_set = np.amax(q_future_by_target[0])

            # map the maximized future reward to current reward
            predicted_next_rewards = self.model_eval.predict(state)
            predicted_next_rewards[0][action] = reward + (1-done) * self.gamma * q_to_set

            # model takes state as input and fits for Y being target_future
            self.model_eval.fit(state, predicted_next_rewards, epochs=1, verbose=0)
    
    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    # since this model only takes about 1 hour to train, we just save at end
    def load_weights(self):
        self.model_eval.load_weights(self.weights_path)
        print("Loaded model_eval weights from file:", self.weights_path)

    def save_weights(self):
        self.model_eval.save_weights(self.weights_path)
        print("Saved model_eval weights in file:",self.weights_path)

    def train_agent(self, episodes_total, epsilon_decay, render=False, save_weights=False):
        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay
        
        self.hist_rewards = []
        self.hist_epsilon_values = []
        
        for episode in range(1, episodes_total+1):
            
            done = False
            state = self.env.reset()
            reward_every_episode = 0
            
            while not done:
                
                if render:
                    self.env.render();
                
                action = self.take_action(state)
                next_state, reward, done, info = self.env.step(action)
                reward_every_episode += reward

                self.memorize(state, action, reward, next_state, done)

                state = next_state

            
            print("Episode: %i/%i | Epsilon: %.5f | Reward: %i" % (episode, episodes_total, self.epsilon, reward_every_episode), end="\r")
            if episode > self.start_after:
                self.decay_epsilon()
                self.learn_from_memory()

            self.hist_rewards.append(reward_every_episode)
            self.hist_epsilon_values.append(self.epsilon)
             
            if(episode % self.update_cycle == 0):
                self.model_target.set_weights(self.model_eval.get_weights())
        print()
        print("Model has finished training. Use test_agent() function to test.")
        if save_weights:
            self.save_weights()

    def test_agent(self, runs=30, render=False, load_weights = False):
        self.epsilon = 0
        best_score = 0
        if load_weights:
            self.load_weights()
        for run in range(runs+1):
            
            done = False
            state = self.env.reset()
            cur_score = 0
            while not done:
                if render:
                    self.env.render()
                action = self.take_action(state)
                next_state, reward, done, info = self.env.step(action)
                cur_score += reward
                state = next_state
            print("Run: %i | Reward: %i" %(run, cur_score))
            best_score = best_score if best_score>cur_score else cur_score
        print("Best Score out of %i runs: %i" % (runs, best_score))
