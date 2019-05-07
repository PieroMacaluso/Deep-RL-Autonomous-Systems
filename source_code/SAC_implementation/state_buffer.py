import datetime

import gym
import numpy as np
import torch
from tensorboardX import SummaryWriter

from image_wrapper import ImageWrapper


class StateBuffer:
    def __init__(self, capacity, initial_state):
        self.get_state = self._get_all
        self.capacity = capacity
        # self.c = initial_state.shape[0]
        # self.h = initial_state.shape[1]
        # self.w = initial_state.shape[2]
        self.h = initial_state.shape[0]
        self.w = initial_state.shape[1]
        self.buffer = []
        self.position = 0
        # Fill the buffer with zeros
        for i in range(self.capacity):
            self.buffer.append(None)
            self.buffer[self.position] = initial_state
            self.position = (self.position + 1) % self.capacity
        # # Insert the first state
        # self.buffer[self.position] = initial_state
        # self.position = (self.position + 1) % self.capacity

    def push(self, state_):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = state_
        self.position = (self.position + 1) % self.capacity
        
    def get_tensor(self):
        state_ = None
        pos_i = 0
        for i in range(self.capacity):
            pos = (self.position + i) % self.capacity
            if state_ is None:
                state_ = torch.tensor(self.buffer[pos], dtype=torch.float32).unsqueeze(0)
            else:
                new_state = torch.tensor(self.buffer[pos], dtype=torch.float32).unsqueeze(0)
                state_ = torch.cat((state_, new_state), dim=0)
            pos_i += 1
        return state_
        
    def _get_all(self):
        state_ = np.empty((self.capacity, self.h, self.w))
        pos_i = 0
        for i in range(self.capacity):
            pos = (self.position + i) % self.capacity
            state_[pos_i] = self.buffer[pos]
            pos_i += 1
        return state_
    
    def _get_mean(self):
        state_ = None
        for i in range(self.capacity):
            pos = (self.position + i) % self.capacity
            if state_ is None:
                state_ = self.buffer[pos]
            else:
                state_ += self.buffer[pos]
        return state_ / self.capacity
    
    def _get_diff(self):
        return self.buffer[1] - self.buffer[0]
    
    def __len__(self):
        return len(self.buffer)


if __name__ == '__main__':
    folder = '{}_StateBuffer_{}/'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), "Pendulum-v0")
    writer = SummaryWriter(log_dir='runs/' + folder)
    
    env = ImageWrapper(512, gym.make("Pendulum-v0"))
    old = env.reset()
    state_buffer = StateBuffer(3, old)
    for i in range(199):
        state = state_buffer.get_state()
        # writer.add_images('episode', state_buffer.get_tensor(), i)
        writer.add_images('episode', state_buffer.get_tensor(), i)
        next_state, reward, done, _ = env.step(env.action_space.sample())
        state_buffer.push(next_state)
    
    env.close()
