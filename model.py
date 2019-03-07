import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# DDPG Model
## Actor model
class DDPGActor(nn.Module):
    def __init__(self, state_size, action_size, seed, fc_units=(64,64)):
        
        super(DDPGActor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc_units = fc_units
        fc_units = (state_size,) + fc_units + (action_size,)
        model_list = []
        for i in range(len(fc_units)-1):
            model_list.append(nn.Linear(fc_units[i], fc_units[i+1]))
            if i != len(fc_units) - 2:
                model_list.append(nn.LayerNorm(fc_units[i+1]))    # Layer Normalization to improve parameter Noise
        self.model = nn.ModuleList(model_list)
        
    
    def forward(self, state):                     # actor model, input is state
        x = state
        for i in range(len(self.fc_units)):
            x = F.relu(self.model[2*i](x))
            x = self.model[2*i+1](x)
        out = F.tanh(self.model[-1](x))  # Here we use tanh because continuous actions is between -1, 1
        return out                             # out put is action    

## Critic Model
class DDPGCritic(nn.Module):
    def __init__(self, state_size, action_size, seed, fc_units=(64,64)):
        super(DDPGCritic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc_units = fc_units
        fc_units = (state_size,) + fc_units + (1,)
        model_list = []
        for i in range(len(fc_units)-1):
            if i == 1:
                model_list.append(nn.Linear(fc_units[i]+action_size, fc_units[i+1]))
            else:     
                model_list.append(nn.Linear(fc_units[i], fc_units[i+1]))        
        self.model = nn.ModuleList(model_list)
    
    def forward(self, xs):
        x, a = xs                                    # critic model, input are state and action
        x = F.leaky_relu(self.model[0](x))
        x = F.leaky_relu(self.model[1](torch.cat([x,a],1)))
        for i in range(2, len(self.fc_units)):
            x = F.leaky_relu(self.model[i](x))
        out = self.model[-1](x)
        return out                                   # output is state value


# PPO Model in A2C style


class FCBody(nn.Module):
    def __init__(self, state_size, fc_units=(64,64), fn_activation=F.relu):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(FCBody, self).__init__()
        self.fc_units = fc_units
        fc_units = (state_size,) + fc_units
        model_list = []
        for i in range(len(fc_units)-1):
            model_list.append(nn.Linear(fc_units[i], fc_units[i+1]))
        self.model = nn.ModuleList(model_list)
        self.fn_activation = fn_activation

    def forward(self, state):
        x = state
        for i in range(len(self.fc_units)):
            x = self.fn_activation(self.model[i](x))
        return x
    
class FCBodyWithAction(nn.Module):
    def __init__(self, state_size, action_size, fc_units=(64,64), fn_activation=F.relu):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(FCBodyWithAction, self).__init__()
        self.fc_units = fc_units
        fc_units = (state_size,) + fc_units
        model_list = []
        for i in range(len(fc_units)-1):
            if i == 1:
                model_list.append(nn.Linear(fc_units[i] + action_size, fc_units[i+1]))
            else:
                model_list.append(nn.Linear(fc_units[i], fc_units[i+1]))
        self.model = nn.ModuleList(model_list)
        self.fn_activation = fn_activation

    def forward(self, state, action):
        x = state
        for i in range(len(self.fc_units)):
            if i == 1:
                x = self.fn_activation(self.model[i](torch.cat([x, action], dim=1)))
            else:
                x = self.fn_activation(self.model[i](x))
        return x
                
class GaussianActorCriticNetwork(nn.Module):
    """A2C Model for continuous action space."""

    def __init__(self, state_size, action_size, seed, fc_units_actor=(64,64), fc_units_critic=(64,64)):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(GaussianActorCriticNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.actor_body = FCBody(state_size, fc_units=fc_units_actor, fn_activation=F.relu)
        self.critic_body = FCBody(state_size, fc_units=fc_units_critic, fn_activation=F.leaky_relu)
        self.fc_action = nn.Linear(fc_units_actor[-1], action_size)
        self.fc_critic = nn.Linear(fc_units_critic[-1], 1)
        
        self.std = nn.Parameter(torch.zeros(action_size))
        
    def forward(self, state, action=None):
        """Build a network that maps state -> action values."""
        phi_a = self.actor_body(state)
        if self.training:
            phi_v = self.critic_body(state)
            mean = F.tanh(self.fc_action(phi_a))           # Here we use tanh bacause continuous actions is between -1, 1
            v = self.fc_critic(phi_v)
            dist = torch.distributions.Normal(mean, F.softplus(self.std))
            if action is None:
                action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1).unsqueeze(-1)
            entropy = dist.entropy().sum(-1).unsqueeze(-1)
            return {'a': action,
                    'log_pi_a': log_prob,
                    'ent': entropy,
                    'mean': mean,
                    'v': v}
        else:
            action = F.tanh(self.fc_action(phi_a))          # Here we use tanh because continuous actions is between -1, 1
            return action
    
    
