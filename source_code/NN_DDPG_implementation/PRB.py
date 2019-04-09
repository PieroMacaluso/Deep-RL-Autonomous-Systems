import random

import torch

from utils.data_structures import SumSegmentTree, MinSegmentTree
from utils.hyperparameters import Config

config = Config()


class PrioritizedReplayMemory(object):
    def __init__(self, size, alpha=0.6, beta_start=0.4, beta_frames=100000):
        super(PrioritizedReplayMemory, self).__init__()
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        
        assert alpha >= 0
        self._alpha = alpha
        
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        
        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2
        
        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0
    
    def beta_by_frame(self, frame_idx):
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)
    
    def push(self, data):
        idx = self._next_idx
        
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize
        
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha
    
    def _encode_sample(self, idxes):
        return [self._storage[i] for i in idxes]
    
    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            mass = random.random() * self._it_sum.sum(0, len(self._storage) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res
    
    def sample(self, batch_size):
        idxes = self._sample_proportional(batch_size)
        
        weights = []
        
        # find smallest sampling prob: p_min = smallest priority^alpha / sum of priorities^alpha
        p_min = self._it_min.min() / self._it_sum.sum()
        
        beta = self.beta_by_frame(self.frame)
        self.frame += 1
        
        # max_weight given to smallest prob
        max_weight = (p_min * len(self._storage)) ** (-beta)
        
        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = torch.tensor(weights, device=config.device, dtype=torch.float)
        encoded_sample = self._encode_sample(idxes)
        return encoded_sample, idxes, weights
    
    def update_priorities(self, idxes, priorities):
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = (priority + 1e-5) ** self._alpha
            self._it_min[idx] = (priority + 1e-5) ** self._alpha
            
            self._max_priority = max(self._max_priority, (priority + 1e-5))
