import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO: Add padding in the formula
def conv2d_size_out(size, kernel_size=3, stride=2):
    """
    Number of Linear input connections depends on output of conv2d layers and
    therefore the input image size, so compute it.
    
    :param kernel_size: default 3
    :param stride: default 2
    :return:
    """
    return (size - (kernel_size - 1) - 1) // stride + 1


class ActorCNN(nn.Module):
    
    def __init__(self, num_inputs, num_actions, h, w, hidden_features=16):
        super(ActorCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, hidden_features, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(hidden_features)
        self.conv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(hidden_features)
        self.conv3 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(hidden_features)
        self.conv4 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=2)
        self.bn4 = nn.BatchNorm2d(hidden_features)
        
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(w))))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(h))))
        linear_input_size = convw * convh * hidden_features
        self.head = nn.Linear(linear_input_size, num_actions)
    
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(x.shape[0], -1)
        return self.head(x)
    
    def get_action(self, state, device):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.forward(state)
        return action.detach().cpu().numpy()[0]


class CriticCNN(nn.Module):
    
    def __init__(self, num_inputs, num_actions, h, w, hidden_feature=16):
        super(CriticCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, hidden_feature, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(hidden_feature)
        self.conv2 = nn.Conv2d(hidden_feature, hidden_feature, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(hidden_feature)
        self.conv3 = nn.Conv2d(hidden_feature, hidden_feature, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(hidden_feature)
        self.conv4 = nn.Conv2d(hidden_feature, hidden_feature, kernel_size=3, stride=2)
        self.bn4 = nn.BatchNorm2d(hidden_feature)
        
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(w))))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(h))))
        linear_input_size = convw * convh * hidden_feature + num_actions  # actions
        self.fc1 = nn.Linear(linear_input_size, 1)
    
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x, a):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(x.shape[0], -1)
        a = a.view(a.shape[0], -1)
        x = torch.cat([x, a], dim=1)
        return self.fc1(x)
