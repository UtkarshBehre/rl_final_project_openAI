import gym
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

# PART 2 Advantage Actor-Critic algorithm
class A2CAgent_0c:
    
    def __init__(self, env_name, score_limit):

        # names used to print results later
        self.env_name = env_name
        self.agent_name = "A2C"
        self.env = gym.make(env_name)
        
        # fetching state and actions shape
        self.state_dim = self.env.observation_space.shape[0]
        self.actions_dim = self.env.action_space.n
        
        # defining useful values
        self.lr_actor = 0.001
        self.lr_critic = 0.005
        self.gamma = 0.99
        self.score_limit = score_limit-5
        self.penalty = abs(score_limit)/2
        self.units = 48
        self.hist_rewards = []
        
        # initializing the models
        self.model_actor = self.build_model(last_activation = "softmax", loss="categorical_crossentropy",lr = self.lr_actor, output_dim = self.actions_dim)
        self.model_critic = self.build_model(last_activation = "linear", loss="mse",lr=self.lr_critic, output_dim = 1)
    
    def build_model(self, last_activation, loss, lr, output_dim):
        '''
        used to initialize actor and critic models 
        generates different models based on the parameters passed
        '''
        model = Sequential()
        model.add(Dense(self.units, input_dim=self.state_dim, activation="relu"))
        model.add(Dense(output_dim, input_dim=self.units, activation = last_activation))     
        model.compile(optimizer=Adam(lr=lr), loss = loss)
        return model
    
    def take_action(self, state):
        # reshape state for the model
        state = np.reshape(state,(1, self.state_dim))
        
        predicted_probs = self.model_actor.predict(state)[0]
        action = np.random.choice(self.actions_dim, p = predicted_probs)
        # action is chosen based on the probabilities output of the model
        return action
        
    def train_every_step(self, state, action, reward, done, next_state):

        state = np.reshape(state, [1, self.state_dim])
        next_state = np.reshape(next_state, [1, self.state_dim])
        critic_target = np.zeros((1,1))
    
        # advantages
        actor_target = np.zeros((1,self.actions_dim))

        value = self.model_critic.predict(state)[0]
        next_value = self.model_critic.predict(next_state)[0]

        if done:
            actor_target[0][action] = reward - value
            critic_target[0][0] = reward
        else:
            actor_target[0][action] = reward + self.gamma * (next_value) - value
            critic_target[0][0] = reward + self.gamma*next_value

        self.model_actor.fit(state, actor_target, epochs=1, verbose=0)
        self.model_critic.fit(state, critic_target, epochs=1, verbose=0)
        
    def train_agent(self, total_episodes):
        '''
        Trains untill the avg score of last 10 runs are above score limit defined
        '''
        for episode in range(1, total_episodes+1):
            episode_reward = 0
            done = False
            state = self.env.reset()
            
            while not done:
                self.env.render()
                action = self.take_action(state)
                next_state, reward, done, _ = self.env.step(action)
                
                # adding penalty helped in reaching peak with more stability
                if done and episode_reward < self.score_limit:
                    reward = -self.penalty
                
                self.train_every_step(state,action,reward, done, next_state)
                
                episode_reward += reward
                state = next_state
            
            # re-add the penalty to print and record in graph
            episode_reward = episode_reward if episode_reward >= self.score_limit else episode_reward+self.penalty
            self.hist_rewards.append(episode_reward)

            print("Agent: %s \t Environment: %s \t Episode: %i/%i | Episode Reward: %i" %(self.agent_name, self.env_name, episode, total_episodes, episode_reward),end="\r")
            
            # check if model performed as per the score limit threshold in last 10 runs and stop the training
            if(np.mean(self.hist_rewards[-min(10, len(self.hist_rewards)):]) > self.score_limit):
                print("%s Agent Training ended at episode %i for %s environment as agent performed best in last 10 runs" %(self.agent_name, episode,self.env_name))
                break

    def test_agent(self, total_runs):
        '''
        tests the agent by running it total_runs number of times against the model
        '''
        best_score = -500
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
            