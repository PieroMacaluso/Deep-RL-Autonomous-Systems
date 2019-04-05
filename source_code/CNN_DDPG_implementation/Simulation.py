import time

import torch
from tensorboardX import SummaryWriter

from NN import ActorNN


class Simulation:
    def __init__(self, env, path, net=ActorNN, n_episode=100, max_len=1000):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = net(self.state_dim, self.action_dim).to(self.device)
        self.net.load_state_dict(torch.load(path))
        self.net.eval()
        self.n_episode = n_episode
        self.max_len = max_len
        self.writer = SummaryWriter(comment='ciao')
    
    def simulate(self):
        total_rewards = 0.0
        steps = 0
        for i in range(self.n_episode):
            t = 0
            rewards = 0
            state = self.env.reset()
            while t < self.max_len:
                action = self.net.get_action(state, self.device)
                next_state, reward, done, _ = self.env.step(action)
                # self.env.render()
                state = next_state
                total_rewards += reward
                rewards += reward
                steps += 1
                t += 1
                if done:
                    break
            self.writer.add_scalar('simulation/reward', rewards, i)
            self.writer.add_scalar('simulation/steps', t, i)
        self.writer.add_text('mean/reward', str(total_rewards / self.n_episode))
        self.writer.add_text('mean/steps', str(steps / self.n_episode))
        time.sleep(1)
