import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DDPGActor(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        super(DDPGActor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        
    
    def forward(self, x):
        x1 = F.relu(self.fc1(x))
        x2 = F.relu(self.fc2(x1))
        out = F.tanh(self.fc3(x2))
        return out

class DDPGCritic(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        super(DDPGCritic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
    
    def forward(self, xs):
        x, a = xs
        x1 = F.relu(self.fc1(x))
        x2 = F.relu(self.fc2(torch.cat([x1,a],1)))
        out = self.fc3(x2)
        return out





class FCBody(nn.Module):
    def __init__(self, state_size, fc1_units=64, fc2_units=64, fn_activation=F.relu):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(FCBody, self).__init__()
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fn_activation = fn_activation

    def forward(self, state):
        x = self.fn_activation(self.fc1(state))
        x = self.fn_activation(self.fc2(x))
        return x
    
class FCBodyWithAction(nn.Module):
    def __init__(self, state_size, action_size, fc1_units=64, fc2_units=64, fn_activation=F.relu):
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
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units + action_size, fc2_units)
        self.fn_activation = fn_activation

    def forward(self, state, action):
        x = self.fn_activation(self.fc1(state))
        x = self.fn_activation(self.fc2(torch.cat([x, action], dim=1)))
        return x
        

class GaussianActorCriticNetwork(nn.Module):
    """A2C Model for continuous action space."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
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
        self.actor_body = FCBody(state_size, fc1_units=fc1_units, fc2_units=fc2_units, fn_activation=F.relu)
        self.critic_body = FCBody(state_size, fc1_units=fc1_units, fc2_units=fc2_units, fn_activation=F.relu)
        self.fc_action = nn.Linear(fc2_units, action_size)
        self.fc_critic = nn.Linear(fc2_units, 1)
        
        self.std = nn.Parameter(torch.zeros(action_size))
        
    def forward(self, state, action=None):
        """Build a network that maps state -> action values."""
        phi_a = self.actor_body(state)
        phi_v = self.critic_body(state)
        mean = F.tanh(self.fc_action(phi_a))
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
    

class DeterministicActorCriticNetwork(nn.Module):
    """DDPG Model for continuous action space."""

    def __init__(self, state_size, action_size, seed, alr, clr, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(DeterministicActorCriticNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.actor_body = FCBody(state_size, fc1_units=fc1_units, fc2_units=fc2_units, fn_activation=F.relu)
        self.critic_body = FCBodyWithAction(state_size, action_size, fc1_units=fc1_units, fc2_units=fc2_units, fn_activation=F.relu)
        self.fc_action = nn.Linear(fc2_units, action_size)
        self.fc_critic = nn.Linear(fc2_units, 1)
        self.actor_opt = optim.Adam(list(self.actor_body.parameters()) + list(self.fc_action.parameters()), lr=alr)
        self.critic_opt = optim.Adam(list(self.critic_body.parameters()) + list(self.fc_critic.parameters()), lr=clr)
    
    def forward(self, state):
        action = self.actor(state)
        return action

    def actor(self, state):
        return F.tanh(self.fc_action(self.actor_body(state)))

    def critic(self, s, a):
        return self.fc_critic(self.critic_body(s, a))
      
    
class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
    
#class for dueling Qnetwork
class DuelingQNetwork(nn.Module):
    """ Model for Dueling QNetwork"""
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.value = QNetwork(state_size, 1, seed)
        self.advantage = QNetwork(state_size, action_size, seed)
        self.aggregate = nn.Linear(action_size + 1, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        val = self.value(state)
        adv = self.advantage(state)
        # Aggregating layer
        # Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum A(s,a'))
        adv = adv - adv.mean(1, keepdim=True)
        x = torch.cat((val, adv),1)
        return self.aggregate(x)
    
