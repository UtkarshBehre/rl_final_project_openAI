import gym
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

# PART 1 REINFORCE algorithm
class ReinforceAgent:
    
    def __init__(self, env_name, score_limit):

        # names used to print results later
        self.env_name = env_name
        self.agent_name = "REINFORCE"
        self.env = gym.make(env_name)

        # fetching state and actions shape
        self.state_dim = self.env.observation_space.shape[0]
        self.actions_dim = self.env.action_space.n
        
        # defining useful values
        self.gamma = 0.99
        self.score_limit = score_limit-5
        self.units = 24
        self.hist_rewards = []
        
        # initializing the model
        self.model = self.build_model()
        
    def build_model(self):
        model = Sequential()
        model.add(Dense(self.units, input_dim = self.state_dim, activation="relu"))
        model.add(Dense(self.units, input_dim = self.units, activation = "relu"))
        model.add(Dense(self.actions_dim, input_dim = self.units, activation="softmax"))
        model.compile(loss="sparse_categorical_crossentropy", optimizer="Adam")
        return model
    
    def take_action(self, state):
        # reshape state for the model
        state = np.reshape(state, [1, self.state_dim])

        predicted_probs = self.model.predict(state)[0]
        action = np.random.choice(self.actions_dim, p = predicted_probs)
        # action is chosen based on the probabilities output of the model
        return action
    
    def train_every_episode(self, states, rewards, actions):
        '''
        states, rewards, actions are lists with data for a full episode
        '''
        total_rewards = 0
        discounted_rewards = []
        
        # iterate rewards in reverse order while discounting
        for reward in rewards[::-1]:
            total_rewards = reward + self.gamma * total_rewards
            discounted_rewards.append(total_rewards)
        
        # reverse again so that it is for respective state
        discounted_rewards.reverse()
        discounted_rewards = np.array(discounted_rewards)

        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

        # stacks the states to make a batch passable on the model
        states = np.vstack(states)
        self.model.train_on_batch(states, actions, sample_weight = discounted_rewards)
            
    def train_agent(self, total_episodes):
        '''
        Trains untill the avg score of last 10 runs are above score limit defined
        '''
        for episode in range(1, total_episodes+1):
            state = self.env.reset()
            done = False

            # lists to hold data of all steps for an episode
            states = []
            rewards = []
            actions = []
            episode_reward = 0
            while not done:
                
                action = self.take_action(state)
                new_state, reward, done, _ = self.env.step(action)
                
                states.append(state)
                rewards.append(reward)
                actions.append(action)
                self.env.render()
                episode_reward += reward
                state = new_state

            self.train_every_episode(states, rewards, actions)
            self.hist_rewards.append(episode_reward)
            
            print("Agent: %s \t Environment: %s \t Episode: %i/%i | Episode Reward: %i" %(self.agent_name, self.env_name, episode, total_episodes, episode_reward),end="\r")
            
            # check if model performed as per the score limit threshold in last 10 runs and stop the training
            if(np.mean(self.hist_rewards[-min(10, len(self.hist_rewards)):]) > self.score_limit-10):
                print("%s Agent Training ended at episode %i for %s environment as agent performed best in last 10 runs" %(self.agent_name, episode,self.env_name))
                break

    def test_agent(self, total_runs):
        '''
        tests the agent by running it total_runs number of times against the model
        '''
        best_score = 0
        for run in range(1, total_runs+1):
            run_score = 0
            done = False
            state = self.env.reset()
            
            while not done:
                new_state, reward, done, _ = self.env.step(self.take_action(state))
                run_score += reward
                state = new_state

            print("Agent: %s \t Environment: %s \t Run: %i/%i | Score: %i" % (self.agent_name, self.env_name, run, total_runs, run_score), end="\r")
            best_score = best_score if best_score>run_score else run_score
        
        print("%s Agent's High Score on %s Environment in %i runs: %i " %(self.agent_name, self.env_name, total_runs, best_score))
                