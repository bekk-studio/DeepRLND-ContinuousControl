import numpy as np
import random
from collections import namedtuple, deque
import pickle

from model import GaussianActorCriticNetwork

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

import os

from collections import deque

BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
LR = 5e-4               # learning rate 

LAMBDA = 0.7             # Gae coefficient
PPO_EPSILON = 0.2        # clip epsilon for PPO
ENTROPY_WEIGHT = 0.01    # Entropy coefficient C2
VF_COEFF = 0.1           # VF coefficient C1
GRADIENT_CLIP = 0.5      # Max norm of the gradient
REPEAT_TIME = 20         # Number of time 


class PPOAgent():
    """Agent PPO"""
    """Initialize an Agent object.
        
        Params
        ======
            env (object): environement
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            device (string): Type of device CPU or GPU
            fc1_units (int): number of output of hidden layer 1
            fc2_units (int): number of output of hidden layer 2
        """
    def __init__(self, env, state_size, action_size, seed, device, fc_units_actor=(64, 64), fc_units_critic=(64, 64)):
       
        self.env = env
        self.brain_name = env.brain_names[0]
        
        self.state_size = state_size
        self.action_size = action_size
        env_info = env.reset(train_mode=True)[self.brain_name]
        self.num_agents = len(env_info.agents)
        self.device = device
        
        # A2C-Network
        self.PPOnet = GaussianActorCriticNetwork(state_size, action_size, seed, fc_units_actor=fc_units_actor, fc_units_critic=fc_units_critic).to(device).train()
        self.PPOnet.fc_action.weight.data.uniform_(-3e-3, 3e-3)
        self.optimizer = optim.Adam(self.PPOnet.parameters(), lr=LR, weight_decay=1e-4)
        
        # State Normalizer
        self.Normalizer = RunningNormalizer(state_size)
        # Reward Normalizer
        self.Reward_Normalizer = RunningNormalizer(1)
    
    def states_norm(self, states):
        """ State Normalization.
             1. learn mean and std incrementally
             2. Normalize according to present mean and std parameters

        Params
        ======
            states (array of float): state input
        """
        for s in states:
            self.Normalizer.push(s)
        return np.array([self.Normalizer.normalize(s) for s in states])
    
    def rewards_norm(self, rewards):
        """ Reward Normalization.
             1. learn mean and std incrementally
             2. Normalize according to present mean and std parameters

        Params
        ======
            states (array of float): state input
        """
        for r in rewards:
            self.Reward_Normalizer.push(np.array([r]))
        return np.array([self.Reward_Normalizer.normalize(np.array([r])) for r in rewards]).squeeze(-1)
    
    def random_sample(self, indices, batch_size):
        indices = np.random.permutation(indices)
        batches = indices[:len(indices) // batch_size * batch_size].reshape(-1, batch_size)
        for batch in batches:
            yield batch
        r = len(indices) % batch_size
        if r:
            yield indices[-r:]

    
    def step(self, max_t=1001, repeat_nbr=1, use_gae = False, use_gradient_clip = False, return_trajectories=False):
        ## Here one step is for one full episode
        """
        1. First, collect some trajectories based on some policy πθ, and initialize theta prime θ'= θ
        2. Next, compute the gradient of the clipped surrogate function using the trajectories
        3. Update θ' using gradient ascent θ' ← θ +α∇θLsurclip(θ,θ)
        4. Then we repeat step 2-3 without generating new trajectories. Typically, step 2-3 are only repeated a few times
        """
        env_info = self.env.reset(train_mode=True)[self.brain_name]  # reset the environment
        states = env_info.vector_observations              # get the current states
        states = self.states_norm(states)  # Normalize the observed states
        states = torch.tensor(states, dtype=torch.float32, device=self.device)                   
        storage = Storage()
        
        t_trajectory = 0
        online_rewards = np.zeros(self.num_agents)
        
        # First, collect some trajectories based on some policy πθ,
        with torch.no_grad():
            for t in range(max_t):
                prediction = self.PPOnet(states)
                for k in prediction.keys():
                    prediction[k] = prediction[k].type(torch.float32)
                actions = prediction['a']
                actions = np.clip(actions, -1, 1)
                env_info = self.env.step(actions.detach().numpy())[self.brain_name]        # send the action to the environment
                next_states = env_info.vector_observations     # get the next state
                next_states = self.states_norm(next_states)  # Normalize the observed states
                rewards = env_info.rewards                   # get the reward
                online_rewards += rewards
                rewards = self.rewards_norm(rewards)        # reward normalisation
                dones = env_info.local_done

                storage.add(prediction)
                storage.add({'r': torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(-1),
                             's': torch.tensor(states, dtype=torch.float32, device=self.device)})
                states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
                t_trajectory += 1

                if np.any(dones):                                  # exit loop if episode finished
                    break


            storage.size(t_trajectory)
            storage.placeholder()
            advantages = torch.tensor(np.zeros((self.num_agents, 1)),dtype=torch.float32, device=self.device)
            returns = prediction['v']
            for i in reversed(range(t_trajectory)):
                returns = storage.r[i] + GAMMA * returns
                if not use_gae:
                    advantages = returns - storage.v[i]
                else:
                    if i == t_trajectory - 1:
                        td_error = storage.r[i] - storage.v[i]
                    else:
                        td_error = storage.r[i] + GAMMA * storage.v[i + 1] - storage.v[i]
                    advantages = advantages * LAMBDA * GAMMA + td_error
                storage.adv[i] = advantages
                storage.ret[i] = returns

            states, actions, log_probs_old, returns, advantages = storage.cat(['s', 'a', 'log_pi_a', 'ret', 'adv'])
            advantages = (advantages - advantages.mean()) / advantages.std()
        
        if return_trajectories:
            return states, actions, log_probs_old, returns, advantages
        
        #compute the gradient of the clipped surrogate function, Update θ' using gradient ascent,repeat step 2-3 without generating new trajectories 
        for _ in range(repeat_nbr):
            sampler = self.random_sample(states.size(0), BATCH_SIZE)
            for batch_indices in sampler:
                
                sampled_states = states[batch_indices]
                sampled_actions = actions[batch_indices]
                sampled_log_probs_old = log_probs_old[batch_indices]
                sampled_returns = returns[batch_indices]
                sampled_advantages = advantages[batch_indices]

                
                prediction = self.PPOnet(sampled_states, sampled_actions)
                for k in prediction.keys():
                    prediction[k] = prediction[k].type(torch.float32)
                ratio = (prediction['log_pi_a'] - sampled_log_probs_old).exp()
                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - PPO_EPSILON,
                                          1.0 + PPO_EPSILON) * sampled_advantages
                
                policy_loss = -torch.min(obj, obj_clipped).mean() - ENTROPY_WEIGHT * prediction['ent'].mean()

                value_loss = VF_COEFF * (sampled_returns - prediction['v']).pow(2).mean()

                self.optimizer.zero_grad()
                (policy_loss + value_loss).backward()

                nn.utils.clip_grad_norm_(self.PPOnet.parameters(), GRADIENT_CLIP)
                self.optimizer.step()
        
        return online_rewards.mean()
    
    def learn(self, n_episodes=200, max_t=2000, use_gae = False, save=False, target=40.):
        """
        Params
        ======
            n_episodes (int): maximum number of training episodes
            max_t (int): maximum number of timesteps per episode
            save (boolean): save or not the model
            target (float) : score threshold to consider that the model is completely trained
        """
    
        scores = []                        # list containing scores from each episode
        scores_avg = []                    # list containing average scores over 100 episode
        scores_window = deque(maxlen=100)  # last 100 scores


        for i_episode in range(1, n_episodes+1):


            score = self.step(max_t, REPEAT_TIME, use_gae = use_gae)

            scores_window.append(score)       # save most recent score
            scores.append(score)              # save most recent score
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
            if i_episode % 10 == 0:
                #print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
                scores_avg.append(np.mean(scores_window))

            if np.mean(scores_window)>=target:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
                scores_avg.append(np.mean(scores_window))
                if save:
                    torch.save(self.PPOnet.state_dict(), 'PPO_model.pt')
                    self.Normalizer.save()
                    break
        return scores, scores_avg
        
    def load(self, saved_net_name):
        self.PPOnet.load_state_dict(torch.load(saved_net_name))
        self.Normalizer.load()
        
    
    def play(self):
        env_info = self.env.reset(train_mode=False)[self.brain_name]  # reset the environment
        
        if os.path.isfile('PPO_model.pt'):
            print('saved model file has been found and loaded')
            self.load('PPO_model.pt')
        else:
            print('No saved model file found')
        
        self.PPOnet.eval()
        
        states = env_info.vector_observations               # get the current state        
        scores = np.zeros(self.num_agents)                  # initialize the score
        while True:
            states = np.array([self.Normalizer.normalize(s) for s in states])  # Normalize the state
            states = torch.tensor(states, dtype=torch.float32, device=self.device)
            actions = self.PPOnet(states).cpu().detach().numpy()            # select an action
            env_info = self.env.step(actions)[self.brain_name]                 # send the action to the environment
            rewards = env_info.rewards                                         # get the reward
            scores += rewards                                                  # update the score
            states = env_info.vector_observations                              # get the next state  
            dones = env_info.local_done
            if True in dones:                                                  # exit loop if episode finished
                break                
        return scores

class Storage:
    def __init__(self, keys=None):
        if keys is None:
            keys = []
        keys = keys + ['s', 'a', 'r',
                       'v', 'ent',
                       'adv', 'ret', 'log_pi_a',
                       'mean']
        self.keys = keys
        self.reset()

    def size(self, size):
        self.size = size
    
    def add(self, data):
        for k, v in data.items():
            assert k in self.keys
            getattr(self, k).append(v)

    def placeholder(self):
        for k in self.keys:
            v = getattr(self, k)
            if len(v) == 0:
                setattr(self, k, [None] * self.size)

    def reset(self):
        for key in self.keys:
            setattr(self, key, [])

    def cat(self, keys):
        data = [getattr(self, k)[:self.size] for k in keys]
        return map(lambda x: torch.cat(x, dim=0), data)
    
class RunningNormalizer:

    def __init__(self, dim):
        self.n = 0
        self.old_m = np.zeros(dim)
        self.new_m = np.zeros(dim)
        self.old_s = np.zeros(dim)
        self.new_s = np.zeros(dim)
        self.dim = dim

    def clear(self):
        self.n = 0
        self.old_s = np.zeros(self.dim)
        
    def save(self, name='PPONormalizer'):
        savedict = {'self.n':self.n, 'self.old_m':self.old_m, 'self.new_m':self.new_m, 'self.old_s':self.old_s, 'self.new_s':self.new_s, 'self.dim':self.dim}
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(savedict, f, pickle.HIGHEST_PROTOCOL)

    def load(self, name='PPONormalizer'):
        with open(name + '.pkl', 'rb') as f:
            savedict = pickle.load(f)
        self.n = savedict['self.n']
        self.old_m = savedict['self.old_m']
        self.new_m = savedict['self.new_m']
        self.old_s = savedict['self.old_s']
        self.new_s = savedict['self.new_s']
        self.dim = savedict['self.dim']

    def push(self, x):
        self.n += 1
        assert x.shape == self.old_m.shape
        
        if self.n == 1:
            self.old_m = self.new_m = x
            #self.old_s = 0
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        return self.new_m if self.n else np.zeros(self.dim)

    def variance(self):
        return self.new_s / (self.n - 1) if self.n > 1 else np.zeros(self.dim)

    def standard_deviation(self):
        return np.sqrt(self.variance())

    def normalize(self, x):
        if self.n <= 1:
            return x
        else:
            return (x - self.mean()) / np.maximum(self.standard_deviation(), [0.1])
