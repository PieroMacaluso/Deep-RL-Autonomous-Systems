import json
import pickle
import random
import time
from collections import deque, namedtuple

import numpy as np
import torch


class ReplayMemory(object):
    def __init__(self, capacity, seed):
        """
        Args:
            capacity:
        """
        self.capacity = capacity
        self.seed = seed
        self.buffer = deque(maxlen=capacity)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        """
        Args:
            state:
            action:
            reward:
            next_state:
            done:
        """
        # if len(self.buffer) < self.capacity:
        #     self.buffer.append(None)
        experience = self.experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """
        Args:
            batch_size:
        """
        out = random.sample(self.buffer, k=batch_size)
        
        states, actions, rewards, next_states, dones = map(np.stack, zip(*out))
        return states, actions, rewards, next_states, dones
    
    def forget_last(self, num_episode_to_forget):
        """
        Args:
            num_episode_to_forget:
        """
        for i in range(num_episode_to_forget):
            self.buffer.pop()
    
    def __len__(self):
        return len(self.buffer)
    
    def dump(self, fp):
        out = list(self.buffer)
        states = list()
        actions = list()
        rewards = list()
        next_states = list()
        dones = list()
        for item in out:
            states.append(item[0].tolist())
            actions.append(item[1].tolist())
            rewards.append(item[2])
            next_states.append(item[3].tolist())
            dones.append(item[4])
        pickle.dump((states, actions, rewards, next_states, dones), fp)
        pass
    
    def separate_out_data_types(self, experiences):
        """Puts the sampled experience into the correct format for a PyTorch neural network"""
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([int(e.done) for e in experiences if e is not None])).float()
        
        return states, actions, rewards, next_states, dones
    
    def load(self, fp):
        (states, actions, rewards, next_states, dones) = pickle.load(fp)
        self.buffer.clear()
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.push(state, action, reward, next_state, done)
        pass


if __name__ == '__main__':
    
    memory = ReplayMemory(10, 1)
    print(len(memory))
    
    for i in range(15):
        memory.push(np.ones((32, 32)) * i, np.ones((32, 32)) * i, i, np.ones((32, 32)) * i, i)
    print(len(memory))
    memory.forget_last(2)
    print(len(memory))
    ts = time.time()
    with open("memory.pkl", "wb") as pickle_out:
        memory.dump(pickle_out)
    delta = time.time() - ts
    print(round(delta, 2))
    ts = time.time()
    memory2 = ReplayMemory(10, 1)
    with open("memory.pkl", "rb") as pickle_out:
        memory2.load(pickle_out)
    print(round(delta, 2))
    print(len(memory))
    one, two, three, four, five = memory.sample(2)
    for i in range(20050):
        memory.push(np.ones((32, 32)) * i, np.ones((32, 32)) * i, i, np.ones((32, 32)) * i, i)
    for i in range(3):
        print(memory.sample(2))
