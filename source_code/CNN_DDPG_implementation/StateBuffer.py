import random

import numpy as np


class StateBuffer:
    def __init__(self, capacity, initial_state):
        self.capacity = capacity
        self.c = initial_state.shape[0]
        self.h = initial_state.shape[1]
        self.w = initial_state.shape[2]
        self.buffer = []
        self.position = 0
        for i in range(self.capacity):
            self.buffer.append(None)
            self.buffer[self.position] = initial_state
            self.position = (self.position + 1) % self.capacity
    
    def push(self, state):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = state
        self.position = (self.position + 1) % self.capacity
    
    def get_state(self):
        state = np.zeros((self.c, self.h, self.w))
        for i in range(self.capacity):
            state += self.buffer[i]
        return state / self.capacity
        
    def __len__(self):
        return len(self.buffer)
