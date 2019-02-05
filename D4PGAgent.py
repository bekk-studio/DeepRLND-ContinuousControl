import numpy as np
import random
from collections import namedtuple, deque
from copy import copy

from model import DeterministicActorCriticNetwork, DDPGCritic, DDPGActor

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from collections import deque

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-2              # for soft update of target parameters
TAU_STEP = 4000         # for hard update of target parameters
ALR = 1e-4              # actor learning rate
CLR = 1e-3              # critic learning rate

#LAMBDA = 0.7             # Gae coefficient
#PPO_EPSILON = 0.2        # clip epsilon for PPO
#ENTROPY_WEIGHT = 0.01
GRADIENT_CLIP = 0.5
T_UPDATE = 20
N_UPDATE = 30
#NOISE_DECAY = 0.995



"""
   DDPG
   Deep Deterministic Policy Gradient
"""
class DDPG():
    """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
    def __init__(self, env, state_size, action_size, seed, device, fc1_units=128, fc2_units=64, parameter_noise=True, state_normalization=True):
       
        self.env = env
        self.brain_name = env.brain_names[0]
        
        self.state_size = state_size
        self.action_size = action_size
        env_info = env.reset(train_mode=True)[self.brain_name]
        self.num_agents = len(env_info.agents)
        self.device = device
        self.parameter_noise = parameter_noise
        self.state_normalization = state_normalization

        self.states = None
        
        self.criterion = nn.MSELoss()

        # Normalizer
        if self.state_normalization:
            self.Normalizer = RunningNormalizer(state_size)
        
        if self.parameter_noise:
            self.std_noise = LinearSchedule(0.1, 0.001, 200000)
        else:
            self.random_process = OrnsteinUhlenbeckProcess(size=(self.num_agents, action_size), std=LinearSchedule(0.2, 0.01, 50000))
        
        # DDPG-Network
        #self.DDGPnet = DeterministicActorCriticNetwork(state_size, action_size, seed, fc1_units=fc1_units, fc2_units=fc2_units, alr=ALR, clr=CLR).to(device).train()
        #self.DDGPnet_target = DeterministicActorCriticNetwork(state_size, action_size, seed, fc1_units=fc1_units, fc2_units=fc2_units, alr=ALR, clr=CLR).to(device).train()
        self.DDPGActor = DDPGActor(state_size, action_size, seed, fc1_units=fc1_units, fc2_units=fc2_units).to(device).train()
        self.DDPGCritic = DDPGCritic(state_size, action_size, seed, fc1_units=fc1_units, fc2_units=fc2_units).to(device).train()
        self.DDPGActor_target = DDPGActor(state_size, action_size, seed, fc1_units=fc1_units, fc2_units=fc2_units).to(device).train()
        self.DDPGCritic_target = DDPGCritic(state_size, action_size, seed, fc1_units=fc1_units, fc2_units=fc2_units).to(device).train()

        self.Actor_optimizer = optim.Adam(self.DDPGActor.parameters(), lr=ALR)
        self.Critic_optimizer = optim.Adam(self.DDPGCritic.parameters(), lr=CLR)

        # Initialize network weights (θ, w) at random
        # Initialize target weights (θ', w') <- (θ, w)
        self.t_step = 0
        self.hard_update(self.DDPGActor, self.DDPGActor_target, TAU_STEP)
        self.hard_update(self.DDPGCritic, self.DDPGCritic_target, TAU_STEP)



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
        for s in states:
            self.Normalizer.push(s)
        return np.array([self.Normalizer.normalize(s) for s in states])
            
    def step(self, max_t=1001): 
       
        # Initialize a random process N for action exploration
        # Receive initial observation state s1
        if self.states is None:
            if not self.parameter_noise:
                self.random_process.reset_states()
            env_info = self.env.reset(train_mode=True)[self.brain_name]  # reset the environment
            self.states = env_info.vector_observations              # get the current states
            if self.state_normalization:                    # Normalize the observed states
                self.states = self.states_norm(self.states)
            self.online_rewards = np.zeros(self.num_agents)
            # Initialize time step (for updating every steps)
            self.t_step = 0
        
        # Select action according to the current policy and exploration noise    
        states = torch.tensor(self.states, dtype=torch.float32, device=self.device)
        if self.parameter_noise:
            DDPGActor_noisy = copy(self.DDPGActor) 
            
            with torch.no_grad():
                for param in DDPGActor_noisy.parameters():
                    param.add_(torch.randn(param.size()) * self.std_noise())
                actions = DDPGActor_noisy(states).cpu().detach().numpy()
        else:
            actions = self.DDPGActor(states).cpu().detach().numpy()
            actions += self.random_process.sample()
        actions = np.clip(actions, -1, 1)
        
        env_info = self.env.step(actions)[self.brain_name]        # send the action to the environment
        next_states = env_info.vector_observations     # get the next state
        if self.state_normalization:                    # Normalize the observed states
            next_states = self.states_norm(next_states)
        rewards = env_info.rewards                   # get the reward
        self.online_rewards += rewards
        # no reward normalisation
        dones = env_info.local_done
        
        for i in range(self.num_agents):
            self.memory.add(self.states[i], actions[i], rewards[i], next_states[i], dones[i])
        
        self.states = next_states
        self.t_step += 1
        
        episode_dones = False
        if self.t_step >= max_t or True in dones:
            episode_dones = True
        
        if len(self.memory) > BATCH_SIZE and self.t_step % T_UPDATE == 0:
            for _ in range(N_UPDATE):
                experiences = self.memory.sample(self.device)
                states, actions, rewards, next_states, dones = experiences

                a_next = self.DDPGActor_target(next_states)
                q_next = self.DDPGCritic_target((next_states, a_next))
                q_next = GAMMA * q_next * (1 - dones)
                q_next.add_(rewards)
                q_next = q_next.detach()

                self.DDPGCritic.zero_grad()

                q = self.DDPGCritic((states, actions))
                #critic_loss = (q - q_next).pow(2).mul(0.5).sum(-1).mean()
                critic_loss = self.criterion(q, q_next)
                
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.DDPGCritic.parameters(), GRADIENT_CLIP)
                self.Critic_optimizer.step()

                self.DDPGActor.zero_grad()
                self.DDPGCritic.zero_grad()

                actions2 = self.DDPGActor(states)
                policy_loss = -self.DDPGCritic((states.detach(), actions2)).mean()

                
                policy_loss.backward()
                self.Actor_optimizer.step()

                self.soft_update(self.DDPGActor, self.DDPGActor_target, TAU)
                self.soft_update(self.DDPGCritic, self.DDPGCritic_target, TAU)
            
        return episode_dones, self.online_rewards.mean()
     
    def learn(self, n_episodes=200, max_t=2000, save=False, target=40.):
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
                episode_dones, score = self.step(max_t)

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
        self.DDGPnet_local = torch.load(saved_net_name)
        
        

class OrnsteinUhlenbeckProcess():
    def __init__(self, size, std, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = 0
        self.std = std
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.reset_states()

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.std() * np.sqrt(self.dt) * np.random.randn(*self.size)
        self.x_prev = x
        return x

    def reset_states(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)
        
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

