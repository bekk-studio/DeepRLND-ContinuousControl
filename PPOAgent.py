import numpy as np
import random
from collections import namedtuple, deque

from model import GaussianActorCriticNetwork

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from collections import deque

BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
LR = 5e-4               # learning rate 

LAMBDA = 0.7             # Gae coefficient
PPO_EPSILON = 0.2        # clip epsilon for PPO
ENTROPY_WEIGHT = 0.01
GRADIENT_CLIP = 0.5
REPEAT_TIME = 20

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PPOAgent():
    """Agent PPO"""
    """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
    def __init__(self, env, state_size, action_size, seed, device):
       
        self.env = env
        self.brain_name = env.brain_names[0]
        
        self.state_size = state_size
        self.action_size = action_size
        env_info = env.reset(train_mode=True)[self.brain_name]
        self.num_agents = len(env_info.agents)
        self.device = device
        
        # A2C-Network
        self.PPOnet = GaussianActorCriticNetwork(state_size, action_size, seed).to(device).train()
        self.optimizer = optim.Adam(self.PPOnet.parameters(), lr=LR)
    
    def random_sample(self, indices, batch_size):
        indices = np.random.permutation(indices)
        batches = indices[:len(indices) // batch_size * batch_size].reshape(-1, batch_size)
        for batch in batches:
            yield batch
        r = len(indices) % batch_size
        if r:
            yield indices[-r:]

    
    def step(self, max_t=1001, repeat_nbr=1, use_gae = False, use_gradient_clip = False, return_trajectories=False):
        """
        1. First, collect some trajectories based on some policy πθ, and initialize theta prime θ'= θ
        2. Next, compute the gradient of the clipped surrogate function using the trajectories
        3. Update θ' using gradient ascent θ' ← θ +α∇θLsurclip(θ,θ)
        4. Then we repeat step 2-3 without generating new trajectories. Typically, step 2-3 are only repeated a few times
        """
        env_info = self.env.reset(train_mode=True)[self.brain_name]  # reset the environment
        states = env_info.vector_observations              # get the current states
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        # no state normalisation
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
                # next state normalization if normalization has been used previously
                rewards = env_info.rewards                   # get the reward
                online_rewards += rewards
                # no reward normalisation
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

                value_loss = 0.5 * (sampled_returns - prediction['v']).pow(2).mean()

                self.optimizer.zero_grad()
                (policy_loss + value_loss).backward()
                if use_gradient_clip:
                    nn.utils.clip_grad_norm_(self.PPOnet.parameters(), GRADIENT_CLIP)
                self.optimizer.step()
        
        return online_rewards.mean()
    
    def learn(self, n_episodes=200, max_t=2000, use_gae = False, use_gradient_clip = False, save=False, target=40.):
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


            score = self.step(max_t, REPEAT_TIME, use_gae = use_gae, use_gradient_clip = use_gradient_clip)

            scores_window.append(score)       # save most recent score
            scores.append(score)              # save most recent score
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
            if i_episode % 10 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
                scores_avg.append(np.mean(scores_window))
                if save:
                    torch.save(self.PPOnet.state_dict(), 'checkpointep{}.pt'.format(i_episode))
            if np.mean(scores_window)>=target:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
                scores_avg.append(np.mean(scores_window))
                if save:
                    torch.save(self.PPOnet.state_dict(), 'model.pt')
                    break
        return scores, scores_avg
        
    def load(saved_net_name):
        self.PPOnet = torch.load(saved_net_name)
    

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
    
