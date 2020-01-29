import torch
import torch.nn as nn
import torch.nn.functional as F


def conv2d_size_out(size, data):
    """
    Number of Linear input connections depends on output of conv2d layers and
    therefore the input image size, so compute it.
    :param size:
    :param kernel_size: default 3
    :param stride: default 2
    :param padding: default 0
    :return:
    """
    res = size
    for layer_name in data:
        layer = data[layer_name]
        # (size - features +2 * padding) // stride +1
        res = (res - layer[2] + 2 * layer[4]) // layer[3] + 1
    return int(res)


def convolutional(data):
    layer = nn.Conv2d(data[0], data[1], kernel_size=data[2], stride=data[3], padding=data[4])
    norm = nn.BatchNorm2d(data[1])
    weights_init_(layer)
    return layer, norm


def linear(data):
    layer = nn.Linear(data[0], data[1])
    weights_init_(layer)
    
    return layer


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ActorCNN(nn.Module):
    
    def __init__(self, num_channel, num_actions, h, w, hidden_features=16):
        super(ActorCNN, self).__init__()
        conv = {
            # 0:fin, 1:fout, 2:kernel, 3:stride, 4:padding
            'conv1': [num_channel, 16, 3, 2, 0],
            'conv2': [16, 16, 3, 2, 0],
            'conv3': [16, 16, 3, 2, 0],
            # 'conv4': [16, 16, 3, 2, 0],
        }
        self.conv_layers_1 = []
        self.bn_layers_1 = []

        self.c1, self.b1 = convolutional(conv['conv1'])
        self.conv_layers_1.append(self.c1)
        self.bn_layers_1.append(self.b1)
        self.c2, self.b2 = convolutional(conv['conv2'])
        self.conv_layers_1.append(self.c2)
        self.bn_layers_1.append(self.b2)
        self.c3, self.b3 = convolutional(conv['conv3'])
        self.conv_layers_1.append(self.c3)
        self.bn_layers_1.append(self.b3)

        self.linear1 = nn.Linear(
            conv2d_size_out(w, conv) * conv2d_size_out(w, conv) * conv['conv3'][1],
            256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, num_actions)
        self.apply(weights_init_)
    
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x1 = x
        for layer, bn in zip(self.conv_layers_1, self.bn_layers_1):
            x1 = F.relu(bn(layer(x1)))
        x = x1.view(x1.shape[0], -1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        return x
    
    def sample(self, state):
        """
        Args:
            state:
        """
        res = self.forward(state)
        return res


class CriticCNN(nn.Module):
    
    def __init__(self, num_channel, num_actions, h, w, hidden_features=16):
        super(CriticCNN, self).__init__()
        conv = {
            # 0:fin, 1:fout, 2:kernel, 3:stride, 4:padding
            'conv1': [num_channel, 16, 3, 2, 0],
            'conv2': [16, 16, 3, 2, 0],
            'conv3': [16, 16, 3, 2, 0],
            # 'conv4': [16, 16, 3, 2, 0],
        }
        self.conv_layers_1 = []
        self.bn_layers_1 = []

        self.c1, self.b1 = convolutional(conv['conv1'])
        self.conv_layers_1.append(self.c1)
        self.bn_layers_1.append(self.b1)
        self.c2, self.b2 = convolutional(conv['conv2'])
        self.conv_layers_1.append(self.c2)
        self.bn_layers_1.append(self.b2)
        self.c3, self.b3 = convolutional(conv['conv3'])
        self.conv_layers_1.append(self.c3)
        self.bn_layers_1.append(self.b3)
        
        self.linear1 = nn.Linear(
            conv2d_size_out(w, conv) * conv2d_size_out(w, conv) * conv['conv3'][1] + num_actions,
            256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 1)
        self.apply(weights_init_)
    
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).

    def forward(self, x, a):
        x1 = x
        for layer, bn in zip(self.conv_layers_1, self.bn_layers_1):
            x1 = F.relu(bn(layer(x1)))
        x = x1.view(x1.shape[0], -1)
        a = a.view(a.shape[0], -1)
        x = torch.cat([x, a], dim=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
