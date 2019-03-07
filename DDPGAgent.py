import numpy as np
import random
from collections import namedtuple, deque
from copy import copy
import pickle

from model import DDPGCritic, DDPGActor

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

import os

from collections import deque

import time

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 64        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
ALR = 1e-4              # actor learning rate
CLR = 5e-4              # critic learning rate


GRADIENT_CLIP = 0.5     # Max norm of gradient
T_UPDATE = 20           # Periode of update step. Update every # time step 
N_UPDATE = 20           # Number of batch update for each update step 
ACTION_NOISE = 0.2      # Standard deviation reference for action noise
PARAM_NOISE = 0.2      # reference for parameter noise
CRITIC_REG = 1e-4       # L2 regularization for critic optimizer


"""
   DDPG
   Deep Deterministic Policy Gradient
"""
class DDPG():
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
            parameter_noise (Boolean): Type of noise, action noise or parameter noise
        """
    def __init__(self, env, state_size, action_size, seed, device, fc_units_act=(64, 64), fc_units_critic=(64, 64), parameter_noise=True, correlated_action_noise = False):
       
        self.env = env
        self.brain_name = env.brain_names[0]
        
        self.state_size = state_size
        self.action_size = action_size
        env_info = env.reset(train_mode=True)[self.brain_name]
        self.num_agents = len(env_info.agents)
        self.device = device
        self.parameter_noise = parameter_noise
        self.correlated_action_noise = correlated_action_noise
     
        self.states = None
        
        # Loss criteron
        self.criterion = nn.MSELoss()
        
        self.distances = []

        # State Normalizer
        self.Normalizer = RunningNormalizer(state_size)
        # Reward Normalizer
        self.Reward_Normalizer = RunningNormalizer(1)
         
        
        # DDPG-Network
        self.DDPGActor = DDPGActor(state_size, action_size, seed, fc_units=fc_units_act).to(device).train()
        self.DDPGCritic = DDPGCritic(state_size, action_size, seed, fc_units=fc_units_critic).to(device).train()
        self.DDPGActor_target = DDPGActor(state_size, action_size, seed, fc_units=fc_units_act).to(device).train()
        self.DDPGCritic_target = DDPGCritic(state_size, action_size, seed, fc_units=fc_units_critic).to(device).train()
        
        if self.parameter_noise:    
            self.perturbed_actor = DDPGActor(state_size, action_size, seed, fc_units=fc_units_act).to(device).train()
            self.adaptative_perturbed_actor = DDPGActor(state_size, action_size, seed, fc_units=fc_units_act).to(device).train()
        
        
        if self.parameter_noise:
            self.param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.1, desired_action_stddev=LinearSchedule(PARAM_NOISE))   #Parameter Noise
        elif self.correlated_action_noise:
            self.random_process = OUNoise(size=(self.num_agents, action_size), seed=seed, sigma=LinearSchedule(ACTION_NOISE))  # Action Noise
        else:
            self.std_noise = LinearSchedule(ACTION_NOISE)

        self.Actor_optimizer = optim.Adam(self.DDPGActor.parameters(), lr=ALR)
        self.Critic_optimizer = optim.Adam(self.DDPGCritic.parameters(), lr=CLR, weight_decay=CRITIC_REG)

        # Initialize network weights (θ, w) at random
        # Initialize target weights (θ', w') <- (θ, w)
        self.t_step = 0
        self.hard_update(self.DDPGActor, self.DDPGActor_target, 1)
        self.hard_update(self.DDPGCritic, self.DDPGCritic_target, 1)

        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
            
    def hard_update(self, local_model, target_model, tau_step):
        """Hard update model parameters. Copy every tau_step

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau_step (int): stop to copy 
        """
        if self.t_step % tau_step == 0: # each tau step
            for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
                target_param.data.copy_(local_param.data)

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
            
    def perturb_actor(self, ref_model, perturb_model, stddev):
        """Perturb actor Network for parameter noise and return action policy

        Params
        ======
            actorToPerturb (PyTorch model): weights will be perturbed from local actor
        """
        self.hard_update(ref_model, perturb_model, 1)
   
        with torch.no_grad():
            for ref_param, perturbed_param in zip(ref_model.parameters(),  perturb_model.named_parameters()):
                if 'ln' not in perturbed_param[0]: 
                    perturbed_param[1].add_(torch.randn(perturbed_param[1].size(),
                                                            dtype=torch.float32, device=self.device) * stddev)

    
    def step(self, max_t=1001): 
        """ learning process for each time step.

        Params
        ======
            max_t (int): max threshold to finish episode 
        """
       
        # Every start of episode
        ## Initialize a random process N for action exploration (parameter Noise or action Noise)
        ## Receive initial observation state s1
        if self.states is None:
            if self.parameter_noise:
                self.perturb_actor(self.DDPGActor, self.perturbed_actor, self.param_noise.current_stddev)
            
            elif self.correlated_action_noise:
                self.random_process.reset()
            
            env_info = self.env.reset(train_mode=True)[self.brain_name]  # reset the environment
            self.states = env_info.vector_observations              # get the current states
                              
            self.states = self.states_norm(self.states) # Normalize the observed states
          
            # Initialize time step (for updating every steps)
            self.t_step = 0
        
        # Select action according to the current policy and exploration noise    
        states = torch.tensor(self.states, dtype=torch.float32, device=self.device)

        if self.parameter_noise:
            
            actions = self.perturbed_actor(states).cpu().detach().numpy()                            

        else:          
            actions = self.DDPGActor(states).cpu().detach().numpy()
            if self.correlated_action_noise:
                actions += self.random_process.sample()
            else:
                actions += np.random.randn(self.num_agents, self.action_size) * self.std_noise()
        
        actions = np.clip(actions, -1, 1)      # Clip noisy actions between -1 and 1
               
        env_info = self.env.step(actions)[self.brain_name]        # send the action to the environment
        next_states = env_info.vector_observations     # get the next state

        next_states = self.states_norm(next_states) # Normalize the observed states
        rewards = env_info.rewards                   # get the reward
        rewards = self.rewards_norm(rewards)        # reward normalisation
        dones = env_info.local_done
        
        # Store step in the replay memory
        for i in range(self.num_agents):
            self.memory.add(self.states[i], actions[i], rewards[i], next_states[i], dones[i])
        
        self.states = next_states
        self.t_step += 1
        
        # Quit the episode if episode are done or time step reach the max treshold
        episode_dones = False
        if self.t_step >= max_t or True in dones:
            episode_dones = True
        
        # Update the model every period of step and when Replay memory is enough large
        if len(self.memory) > BATCH_SIZE and self.t_step % T_UPDATE == 0:
                        
            if self.parameter_noise: ## For adaptation of parameter noise according to update of main actor model
                self.perturb_actor(self.DDPGActor, self.adaptative_perturbed_actor, self.param_noise.current_stddev) # Add parameter noise to dedicated model                
                self.param_noise.adapt(self.adaptative_perturbed_actor(states).cpu().detach().numpy(), self.DDPGActor(states).cpu().detach().numpy()) # Adapt standard deviation                          
                self.distances.append(self.param_noise.distance)   # for debug
                       
            for _ in range(N_UPDATE):   ## Update the model several time by sampling a batches from replay memory
                experiences = self.memory.sample(self.device)
                states, actions, rewards, next_states, dones = experiences

                # Update the critic model
                a_next = self.DDPGActor_target(next_states)
                q_next = self.DDPGCritic_target((next_states, a_next))
                q_next = GAMMA * q_next * (1 - dones)
                q_next.add_(rewards)
                q_next = q_next.detach()

                self.DDPGCritic.zero_grad()

                q = self.DDPGCritic((states, actions))
                critic_loss = self.criterion(q, q_next)
                
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.DDPGCritic.parameters(), GRADIENT_CLIP)
                self.Critic_optimizer.step()

                # Update the actor model
                self.DDPGActor.zero_grad()
                self.DDPGCritic.zero_grad()

                actions2 = self.DDPGActor(states)
                policy_loss = -self.DDPGCritic((states.detach(), actions2)).mean()

                policy_loss.backward()
                self.Actor_optimizer.step()

                self.soft_update(self.DDPGActor, self.DDPGActor_target, TAU)
                self.soft_update(self.DDPGCritic, self.DDPGCritic_target, TAU)
            
        return episode_dones   
     
    def evaluate(self):
        
        env_info = self.env.reset(train_mode=True)[self.brain_name]  # reset the environment
        states = env_info.vector_observations            
        online_rewards = np.array(env_info.rewards)        
        episode_dones = False
        while not episode_dones:
            states = np.array([self.Normalizer.normalize(s) for s in states])  # Normalize the observed states
            states = torch.tensor(states, dtype=torch.float32, device=self.device)
            actions = self.DDPGActor(states).cpu().detach().numpy()
            env_info = self.env.step(actions)[self.brain_name]        # send the action to the environment
            rewards = env_info.rewards                   # get the reward
            online_rewards += rewards
            states = env_info.vector_observations     # get the next state  
            dones = env_info.local_done
            if True in dones:
                episode_dones = True
                
        return online_rewards.mean()
    
    def learn(self, n_episodes=200, max_t=2000, save=False, target=10000.):
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

            self.states = None
            episode_dones = False

            while not episode_dones:                
                episode_dones = self.step(max_t)

            score = self.evaluate()           
            
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
                    torch.save(self.DDPGActor.state_dict(), 'DDPGActor_model.pt')
                    self.Normalizer.save()
                    break
        return scores, scores_avg
        
    def load(self, saved_net_name):
        self.DDPGActor.load_state_dict(torch.load(saved_net_name))
        self.Normalizer.load()
        
    def play(self):
        env_info = self.env.reset(train_mode=False)[self.brain_name]  # reset the environment
        
        if os.path.isfile('DDPGActor_model.pt'):
            print('saved model file has been found and loaded')
            self.load('DDPGActor_model.pt')
        else:
            print('No saved model file found')
        
        self.DDPGActor.eval()
        
        states = env_info.vector_observations               # get the current state        
        scores = np.zeros(self.num_agents)                  # initialize the score
        while True:
            states = np.array([self.Normalizer.normalize(s) for s in states])  # Normalize the state
            states = torch.tensor(states, dtype=torch.float32, device=self.device)
            actions = self.DDPGActor(states).cpu().detach().numpy()            # select an action
            env_info = self.env.step(actions)[self.brain_name]                 # send the action to the environment
            rewards = env_info.rewards                                         # get the reward
            scores += rewards                                                  # update the score
            states = env_info.vector_observations                              # get the next state  
            dones = env_info.local_done
            if True in dones:                                                  # exit loop if episode finished
                break

        return scores

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.size = size
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma() * np.random.randn(*self.size)
        self.state = x + dx
        return self.state
        
class LinearSchedule:
    def __init__(self, start, end=None, steps=None):
        if end is None:
            end = start
            steps = 1
        self.inc = (end - start) / float(steps)
        self.current = start
        self.end = end
        if end > start:
            self.bound = min
        else:
            self.bound = max

    def __call__(self, steps=1):
        val = self.current
        self.current = self.bound(self.current + self.inc * steps, self.end)
        return val

    
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self, device):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class RunningNormalizer:

    def __init__(self, dim):
        self.n = 0
        self.old_m = np.zeros(dim)
        self.new_m = np.zeros(dim)
        self.old_s = np.zeros(dim)
        self.new_s = np.zeros(dim)
        self.dim = dim

    def save(self, name='DDPGNormalizer'):
        savedict = {'self.n':self.n, 'self.old_m':self.old_m, 'self.new_m':self.new_m, 'self.old_s':self.old_s, 'self.new_s':self.new_s, 'self.dim':self.dim}
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(savedict, f, pickle.HIGHEST_PROTOCOL)

    def load(self, name='DDPGNormalizer'):
        with open(name + '.pkl', 'rb') as f:
            savedict = pickle.load(f)
        self.n = savedict['self.n']
        self.old_m = savedict['self.old_m']
        self.new_m = savedict['self.new_m']
        self.old_s = savedict['self.old_s']
        self.new_s = savedict['self.new_s']
        self.dim = savedict['self.dim']
    
    def clear(self):
        self.n = 0
        self.old_s = np.zeros(self.dim)

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
        
class AdaptiveParamNoiseSpec:
    def __init__(self, initial_stddev=0.1, desired_action_stddev=0.1, adoption_coefficient=1.01):
        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adoption_coefficient = adoption_coefficient
        self.current_stddev = initial_stddev
       
    def adapt(self, actions, perturbed_actions):
        distance = np.sqrt(np.mean(np.square(actions - perturbed_actions)))
        self.distance = distance
        if type(self.desired_action_stddev) == float:
            threshold = self.desired_action_stddev
        else:
            threshold = self.desired_action_stddev()
        if distance > threshold:
            # Decrease stddev.
            self.current_stddev /= self.adoption_coefficient
        else:
            # Increase stddev.
            self.current_stddev *= self.adoption_coefficient



