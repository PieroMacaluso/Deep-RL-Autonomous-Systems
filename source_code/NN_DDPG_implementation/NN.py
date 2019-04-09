import torch
import torch.nn.functional as F
from torch import nn


class CriticNN(nn.Module):
    """
    Value Network
    """
    
    def __init__(self, num_inputs, num_actions, hidden_size=256, init_w=3e-3):
        super(CriticNN, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
    
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class ActorNN(nn.Module):
    """
    Policy Network
    """
    
    def __init__(self, num_inputs, num_actions, hidden_size=256, init_w=3e-3):
        super(ActorNN, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_actions)
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = nn.Tanh(self.linear3(x))
        return x
