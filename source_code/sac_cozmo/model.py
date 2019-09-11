import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6
IMAGE_H = 64
IMAGE_W = 64


def conv2d_size_out(size, data):
    """Number of Linear input connections depends on output of conv2d layers and
    therefore the input image size, so compute it. Data is a dictionary. :param
    size: size of the input of the convolution :param data: details about
    convolutional layers. :return:

    Args:
        size:
        data:
    """
    """
    
    :param size:
    :param kernel_size: default 3
    :param stride: default 2
    :param padding: default 0
    :return:
    """
    """
    """
    res = size
    for layer_name in data:
        layer = data[layer_name]
        # (size - features +2 * padding) // stride +1
        res = (res - layer[2] + 2 * layer[4]) // layer[3] + 1
    return int(res)


# Initialize Policy weights
def weights_init_(m):
    """
    Args:
        m:
    """
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


def convolutional(data):
    """
    Args:
        data:
    """
    layer = nn.Conv2d(data[0], data[1], kernel_size=data[2], stride=data[3], padding=data[4])
    norm = nn.BatchNorm2d(data[1])
    weights_init_(layer)
    return layer, norm


'''CONVOLUTIONAL NEURAL NETWORKS'''


class ValueNetworkCNN(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        """
        Args:
            num_inputs:
            hidden_dim:
        """
        super(ValueNetworkCNN, self).__init__()
        
        conv = {
            # 0:fin, 1:fout, 2:kernel, 3:stride, 4:padding
            'conv1': [num_inputs, 16, 8, 4, 0],
            'conv2': [16, 32, 4, 2, 0]
        }
        
        self.conv1, self.bn1 = convolutional(conv['conv1'])
        self.conv2, self.bn2 = convolutional(conv['conv2'])
        
        self.linear1 = nn.Linear(conv2d_size_out(IMAGE_H, conv) * conv2d_size_out(IMAGE_W, conv) * conv['conv2'][1], hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        
        self.apply(weights_init_)
    
    def forward(self, state):
        """
        Args:
            state:
        """
        x = F.relu(self.bn1(self.conv1(state)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QNetworkCNN(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        """
        Args:
            num_inputs:
            num_actions:
            hidden_dim:
        """
        super(QNetworkCNN, self).__init__()
        
        conv = {
            # 0:fin, 1:fout, 2:kernel, 3:stride, 4:padding
            'conv1': [num_inputs, 16, 8, 4, 0],
            'conv2': [16, 32, 4, 2, 0]
        }
        
        self.conv1, self.bn1 = convolutional(conv['conv1'])
        self.conv2, self.bn2 = convolutional(conv['conv2'])
        self.conv3, self.bn3 = convolutional(conv['conv1'])
        self.conv4, self.bn4 = convolutional(conv['conv2'])
        
        # Q1 architecture
        self.linear1 = nn.Linear(conv2d_size_out(IMAGE_H, conv) * conv2d_size_out(IMAGE_W, conv) * conv['conv2'][1] + num_actions,
                                 hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        
        # Q2 architecture
        self.linear4 = nn.Linear(conv2d_size_out(IMAGE_H, conv) * conv2d_size_out(IMAGE_W, conv) * conv['conv2'][1] + num_actions,
                                 hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)
        
        self.apply(weights_init_)
    
    def forward(self, state, action):
        """
        Args:
            state:
            action:
        """
        x1 = F.relu(self.bn1(self.conv1(state)))
        x1 = F.relu(self.bn2(self.conv2(x1)))
        x1 = x1.view(x1.shape[0], -1)
        xu1 = torch.cat([x1, action], 1)
        
        x1 = F.relu(self.linear1(xu1))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)
        
        x2 = F.relu(self.bn3(self.conv3(state)))
        x2 = F.relu(self.bn4(self.conv4(x2)))
        x2 = x2.view(x2.shape[0], -1)
        xu2 = torch.cat([x2, action], 1)
        x2 = F.relu(self.linear4(xu2))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)
        
        return x1, x2


class GaussianPolicyCNN(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        """
        Args:
            num_inputs:
            num_actions:
            hidden_dim:
        """
        super(GaussianPolicyCNN, self).__init__()
        
        conv = {
            # 0:fin, 1:fout, 2:kernel, 3:stride, 4:padding
            'conv1': [num_inputs, 16, 8, 4, 0],
            'conv2': [16, 32, 4, 2, 0]
        }
        
        self.conv1, self.bn1 = convolutional(conv['conv1'])
        self.conv2, self.bn2 = convolutional(conv['conv2'])
        
        self.linear1 = nn.Linear(conv2d_size_out(IMAGE_H, conv) * conv2d_size_out(IMAGE_W, conv) * conv['conv2'][1], hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)
        
        self.apply(weights_init_)
    
    def forward(self, state):
        """
        Args:
            state:
        """
        x = F.relu(self.bn1(self.conv1(state)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std
    
    def sample(self, state):
        """
        Args:
            state:
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, torch.tanh(mean)


class DeterministicPolicyCNN(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        """
        Args:
            num_inputs:
            num_actions:
            hidden_dim:
        """
        super(DeterministicPolicyCNN, self).__init__()
        conv = {
            # 0:fin, 1:fout, 2:kernel, 3:stride, 4:padding
            'conv1': [num_inputs, 16, 8, 4, 0],
            'conv2': [16, 32, 4, 2, 0]
        }
        
        self.conv1, self.bn1 = convolutional(conv['conv1'])
        self.conv2, self.bn2 = convolutional(conv['conv2'])
        
        self.linear1 = nn.Linear(conv2d_size_out(IMAGE_H, conv) * conv2d_size_out(IMAGE_W, conv) * conv['conv2'][1], hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mean = nn.Linear(hidden_dim, num_actions)
        # TODO: fix this
        self.noise = torch.Tensor(num_actions).to("cuda:0")
        
        self.apply(weights_init_)
    
    def forward(self, state):
        """
        Args:
            state:
        """
        x = F.relu(self.bn1(self.conv1(state)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x))
        return mean
    
    def sample(self, state):
        """
        Args:
            state:
        """
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean


'''NEURAL NETWORKS'''


class ValueNetworkNN(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        """
        Args:
            num_inputs:
            hidden_dim:
        """
        super(ValueNetworkNN, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        
        self.apply(weights_init_)
    
    def forward(self, state):
        """
        Args:
            state:
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QNetworkNN(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        """
        Args:
            num_inputs:
            num_actions:
            hidden_dim:
        """
        super(QNetworkNN, self).__init__()
        
        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        
        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)
        
        self.apply(weights_init_)
    
    def forward(self, state, action):
        """
        Args:
            state:
            action:
        """
        xu = torch.cat([state, action], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)
        
        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)
        
        return x1, x2


class GaussianPolicyNN(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        """
        Args:
            num_inputs:
            num_actions:
            hidden_dim:
        """
        super(GaussianPolicyNN, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)
        
        self.apply(weights_init_)
    
    def forward(self, state):
        """
        Args:
            state:
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std
    
    def sample(self, state):
        """
        Args:
            state:
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, torch.tanh(mean)


class DeterministicPolicyNN(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        """
        Args:
            num_inputs:
            num_actions:
            hidden_dim:
        """
        super(DeterministicPolicyNN, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)
    
    def forward(self, state):
        """
        Args:
            state:
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x))
        return mean
    
    def sample(self, state):
        """
        Args:
            state:
        """
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean
