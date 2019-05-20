import random
from functools import partial
from operator import is_not

import numpy as np


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def push_memory(self, episode_memory):
        for i_replay in episode_memory.buffer:
            if len(self.buffer) < self.capacity:
                self.buffer.append(None)
            self.buffer[self.position] = i_replay
            self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample([x for x in self.buffer if x is not None], batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def forget_last(self, num_episode_to_forget):
        for i in range(num_episode_to_forget):
            self.position = (self.position - 1) % self.capacity
            if self.position < 0:
                self.position += self.capacity
            self.buffer[self.position] = None
    
    def __len__(self):
        return len(self.buffer)


if __name__ == '__main__':
    memory = ReplayMemory(10000)
    for i in range(20050):
        memory.push(i,i,i,i,i)
    memory.forget_last(256)
    batch = memory.sample(256)
    assert [x for x in batch if x is not None]
    for i in range(20050):
        memory.push(i,i,i,i,i)

