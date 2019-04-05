import torch
from torch import nn
import torch.nn.functional as F


class CriticCNN(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(CriticCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 16, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, stride=2)
        self.bn4 = nn.BatchNorm2d(16)
    
        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=3, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
    
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(w))))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(h))))
        linear_input_size = convw * convh * 16 + num_actions  # actions
        self.fc1 = nn.Linear(linear_input_size, 1)
    
    def forward(self, state, action):
        x = F.relu(self.bn1(self.conv1(state)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(x.shape[0], -1)
        a = action.view(action.shape[0], -1)
        x = torch.cat([x, a], dim=1)
        return self.fc1(x)


class ActorCNN(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(ActorCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 16, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, stride=2)
        self.bn4 = nn.BatchNorm2d(16)
    
        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=3, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
    
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(w))))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(h))))
        linear_input_size = convw * convh * 16
        self.head = nn.Linear(linear_input_size, num_actions)
    
    def forward(self, state):
        x = F.relu(self.bn1(self.conv1(state)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(x.shape[0], -1)
        return self.head(x)
    
    def get_action(self, state, device):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.forward(state)
        return action.detach().cpu().numpy()[0, 0]