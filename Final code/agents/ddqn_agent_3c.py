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

class DDQNAgent_3c:
    def __init__(self, env, memory_len = 10000):
        
        self.env = env
        self.agent_name = "DDQN_Agent_3c"

        self.state_dim = self.env.observation_space.shape
        self.actions_dim = self.env.action_space.n
        
        self.n_features = np.prod(np.array(self.state_dim))
        
        self.memory_len = memory_len
        self.batch_size = 32

        self.learning_rate = 0.0001
        self.gamma = 0.99
        
        # using np array so that calculations are faster instead of dequeue
        self.memory = np.zeros((self.memory_len, 3 + self.n_features*2)) 

        self.epsilon = 1.0
        self.epsilon_min = 0.05


        self.save_weights_cycle = 100
        self.start_after = 10
        self.update_cycle = 50
        self.learn_cycle = 1

        self.memory_counter = 0
        
        self.model_eval = self.build_model() 
        self.model_target = self.build_model()

        print("===============================")
        print("The agent",self.agent_name,"is initiallized with below parameter values.")
        print("memory buffer limit:%i\nbatch size:%i\nlearning_rate:%.4f\ngamma:%.2f\n" % (self.memory_len,self.batch_size,self.learning_rate, self.gamma))
        print("Below is the neural network used:")
        print(self.model_eval.summary())
        print("===============================")

        self.model_eval_path = "model_eval_weights_"
        self.model_target_path = "model_target_weights_"
        
    def build_model(self):

        kernel_initializer = keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution="normal", seed=None)
        bias_initializer = "zeros"
        
        model = Sequential()
        
        model.add(Conv2D(input_shape=self.state_dim,data_format="channels_last",
            filters=32, kernel_size=(8,8), strides=(4,4),
        padding="same", activation="relu",
            kernel_initializer=kernel_initializer))
        
        model.add(Conv2D(data_format="channels_last",
            filters=64, kernel_size=(4,4), strides=(2,2),
        padding="same", activation="relu",
            kernel_initializer=kernel_initializer))
        
        model.add(Conv2D(data_format="channels_last",
            filters=64, kernel_size=(3,3), strides=(1,1),
        padding="same", activation="relu",
            kernel_initializer=kernel_initializer))
        
        model.add(Flatten(data_format="channels_last"))
        
        model.add(Dense(units=512, activation="relu",
            kernel_initializer=kernel_initializer,
            bias_initializer = bias_initializer))
        
        # None activation is same as linear activation
        model.add(Dense(units=self.actions_dim, activation=None,
            kernel_initializer=kernel_initializer,
            bias_initializer = bias_initializer))
        
        optimizer = keras.optimizers.RMSprop(learning_rate=self.learning_rate, rho=0.99)
        
        model.compile(loss="mse", optimizer=optimizer)
        
        return model
    

    def memorize(self, state, action, reward, done, next_state):

        transition = np.hstack((np.reshape(state, [-1]), [action, reward, int(done)], np.reshape(next_state, [-1])))
        index = self.memory_counter % self.memory_len 
        self.memory[index,:] = transition
        self.memory_counter +=1
    
    def learn_from_memory(self):

        if self.memory_counter > self.memory_len: 
            sample_indices = np.random.choice(self.memory_len, size=self.batch_size, replace=False)
        else: 
            sample_indices = np.random.choice(self.memory_counter, size=self.batch_size)

        minibatch = self.memory[sample_indices, :]
        states_shape = [self.batch_size] + [dim for dim in self.state_dim] # [batch_size, 84, 84, 4]
        
        states = np.reshape(minibatch[:,:self.n_features], newshape=states_shape)
        actions = minibatch[:,self.n_features].astype(int)
        rewards = minibatch[:,self.n_features +1]
        done = minibatch[:, self.n_features + 2]
        next_states = np.reshape(minibatch[:,-self.n_features:], newshape=states_shape)
        
        q_new = self.model_eval.predict(states)
        q_old = self.model_target.predict(next_states)
        q_target = q_new.copy() 
        
        # gets list of indices 0 to batchsize
        batch_indices = np.arange(self.batch_size, dtype=np.int32) 

        q_new_future = self.model_eval.predict(next_states)
        best_actions_future = np.argmax(q_new_future, axis=1)
        selected_q_old = q_old[batch_indices, best_actions_future]

        q_target[batch_indices,actions] = rewards + (1-done) * self.gamma * selected_q_old
        
        self.model_eval.fit(states, q_target, epochs=1, verbose=0)
    

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train_agent(self, total_episodes=18000, epsilon_decay=0.9998, render = False, save_weights=False):
        print("Now training", self.agent_name)

        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay

        self.hist_rewards = []
        self.hist_epsilon_values = []


        start_time = time.time()
        for episode in range(1, total_episodes+1):
            
            done = False
            state = self.env.reset()
            reward_every_episode = 0
            
            while not done:

                if render:
                    self.env.render()

                action = self.take_action(state) 
                next_state, reward, done, info = self.env.step(action)
                reward_every_episode += reward
                
                self.memorize(state, action, reward, done, next_state)
                
                state = next_state

            if episode > self.start_after:
                self.epsilon = self.epsilon*self.epsilon_decay
                if episode % self.learn_cycle == 0:
                    self.learn_from_memory()
            
                if episode % self.update_cycle == 0:
                    self.model_target.set_weights(self.model_eval.get_weights())
            
            self.hist_rewards.append(reward_every_episode)
            self.hist_epsilon_values.append(self.epsilon)

            if save_weights and (episode % self.save_weights_cycle == 0):
                self.model_eval.save_weights(self.model_eval_path + str(episode) +".h5")
                self.model_target.save_weights(self.model_target_path + str(episode) +".h5")

            time_spent = time.time() - start_time
            print('Episode: %i/%i, Episode Reward: %i, Epsilon: %.5f, Time Spent: %i hours %i minutes' % 
                (episode, total_episodes, reward_every_episode, self.epsilon , int((time_spent/60)/60), int((time_spent/60)%60)), end="\r")
        print("\nTraining of", self.agent_name, "completed.")
        
    def take_action(self, state, test=True):   
        if (np.random.uniform() <= self.epsilon):
            return np.random.randint(0, self.actions_dim)
        else:
            state = np.expand_dims(state, axis=0)
            actions_value = self.model_eval.predict(state)
            return np.argmax(actions_value)
    
    def test_agent(self, runs = 30, render = False):
        
        self.epsilon = 0
        best_score = 0

        for run in range(1, runs+1):

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
                print("Agent: %s \t Run: %i/%i | Score: %i" % (self.agent_name, run, runs, cur_score), end="\r")
            
            best_score = best_score if best_score>cur_score else cur_score

        print("\n%s Agent's High Score in %i runs: %i " %(self.agent_name, runs, best_score))
        
    def set_model_weights(self, e_num):
        self.model_eval.load_weights(self.model_eval_path+str(e_num)+".h5")
        self.model_target.load_weights(self.model_target_path+str(e_num)+".h5")
        
    def test_runs_progressive(self, e_nums, episodes = 10, rend = True):
        for e_num in e_nums:
            print("weights at:",e_num)
            self.set_model_weights(e_num)
            self.test_agent()