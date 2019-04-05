import argparse

import gym
import matplotlib.pyplot as plt
import torch

from DDPG import DDPG
from NormalizedActions import NormalizedActions
from OUNoise import OUNoise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='State implementation of DDPG.')
    parser.add_argument('env', metavar='env_name', type=str, nargs='+',
                        help='Name of the Gym Environment')
    args = parser.parse_args()
    
    plt.ion()
    
    # if gpu is to be used
    assert torch.cuda.is_available(), "CUDA is not available"
    
    env = NormalizedActions(gym.make(args.env[0]))
    test_env = NormalizedActions(gym.make(args.env[0]))
    ou_noise = OUNoise(env.action_space)
    
    ddpg = DDPG(env, test_env, exp_strategy=ou_noise)
    ddpg.train()
    env.close()
