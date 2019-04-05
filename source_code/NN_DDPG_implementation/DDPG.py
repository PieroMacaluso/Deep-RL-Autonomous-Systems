import os
import time

import gym
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import optim, nn

from NN import ActorNN, CriticNN
from OUNoise import OUNoise
from ReplayBuffer import ReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DDPG:
    """
    Deep Deterministic Policy Gradient.
    """
    # TODO: Check Epsilon presence in DDPG
    # TODO: Implement Evaluation of the Policy
    # TODO: Implement Checkpoint of the Policy
    
    EPSILON_START = 0.99
    EPSILON_END = 0.05
    EPSILON_DECAY = 500
    
    def __init__(
            self,
            env: gym.Env,
            test_env: gym.Env,
            actor_nn=ActorNN,
            critic_nn=CriticNN,
            exp_strategy=OUNoise,
            eps_start=0.9,
            eps_end=0.2,
            eps_decay=1000,
            batch_size=32,
            n_episode=1000,
            episode_max_len=1000,
            replay_min_size=10000,
            replay_max_size=1000000,
            discount=0.99,
            critic_weight_decay=0.,
            critic_update_method='adam',
            critic_lr=1e-3,
            actor_weight_decay=0,
            actor_update_method='adam',
            actor_lr=1e-4,
            eval_samples=10000,
            soft_target_tau=0.001,
            n_updates_per_sample=1,
            checkpoint_dir='./checkpoints/'):
        """
        DDPG constructor
        
        :param env: Environment.
        :param actor_nn: Actor (Policy) NN.
        :param critic_nn: Critic (Value) NN.
        :param exp_strategy: Exploration strategy.
        :param batch_size: Number of samples for each minibatch.
        :param n_episode: Number of Episode.
        :param episode_max_len: How many timesteps for each Episode.
        :param replay_min_size: Minimum size of the replay buffer to start training.
        :param replay_max_size: Size of the experience replay pool.
        :param discount: Discount factor (Gamma) for the cumulative return.
        :param critic_weight_decay: Weight decay factor for parameters of the Q function.
        :param critic_update_method: Online optimization method for training Q function.
        :param critic_lr: Learning rate for training Q function.
        :param actor_weight_decay: Weight decay factor for parameters of the policy.
        :param actor_update_method: Online optimization method for training the policy.
        :param actor_lr: Learning rate for training the policy.
        :param eval_samples: Number of samples (timesteps) for evaluating the policy.
        :param soft_target_tau: Interpolation parameter for doing the soft target update.
        :param n_updates_per_sample: Number of Q function and policy updates per new sample obtained.
        :param checkpoint_dir: Checkpoint Directory in which we save our best Checkpoint
        """
        
        self.writer = SummaryWriter()
        self.env = env
        self.test_env = test_env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.noise = exp_strategy
        self.batch_size = batch_size
        self.n_episode = n_episode
        self.episode_max_len = episode_max_len
        self.replay_min_size = replay_min_size
        self.replay_max_size = replay_max_size
        self.replay_buffer = ReplayBuffer(self.replay_max_size)
        self.discount = discount
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        
        self.critic_net = critic_nn(self.state_dim, self.action_dim).to(device)
        self.actor_net = actor_nn(self.state_dim, self.action_dim).to(device)
        
        self.target_value_net = critic_nn(self.state_dim, self.action_dim).to(device)
        self.target_policy_net = actor_nn(self.state_dim, self.action_dim).to(device)
        
        for target_param, param in zip(self.target_value_net.parameters(), self.critic_net.parameters()):
            target_param.data.copy_(param.data)
        
        for target_param, param in zip(self.target_policy_net.parameters(), self.actor_net.parameters()):
            target_param.data.copy_(param.data)
        
        self.critic_opt = optim.Adam(self.critic_net.parameters(), lr=critic_lr)
        self.actor_opt = optim.Adam(self.actor_net.parameters(), lr=actor_lr)
        
        self.critic_lr = critic_lr
        self.critic_weight_decay = critic_weight_decay
        self.actor_lr = actor_lr
        self.actor_weight_decay = actor_weight_decay
        self.critic_loss = nn.MSELoss()
        
        self.eval_samples = eval_samples
        self.soft_target_tau = soft_target_tau
        self.n_updates_per_sample = n_updates_per_sample
        
        self.checkpoint_dir = checkpoint_dir
    
    def test(self, count=10):
        rewards = 0.0
        steps = 0
        for _ in range(count):
            t = 0
            state = self.test_env.reset()
            while t < self.episode_max_len:
                action = self.actor_net.get_action(state, device)
                next_state, reward, done, _ = self.env.step(action)
                # self.env.render()
                state = next_state
                rewards += reward
                steps += 1
                if done:
                    break
        return rewards / count, steps / count
    
    def train(self):
        episode = 0
        running_episode_reward_100 = 0
        running_episode_reward = 0
        rewards = []
        eps = self.eps_start
        frame_idx = 0
        best_reward = None
        while episode < self.n_episode:
            state = self.env.reset()
            # TODO: Check if this reset is OK
            self.noise.reset()
            episode_reward = 0
            upgrade_steps = 0
            running_ploss = 0
            running_vloss = 0
            step = 0
            while step < self.episode_max_len:
                action = self.actor_net.get_action(state, device)
                eps = self.eps_start - (self.eps_start - self.eps_end) * min(1.0, episode / self.eps_decay)
                action = self.noise.get_action(action, eps)
                next_state, reward, done, _ = self.env.step(action)
                # self.env.render()
                
                self.replay_buffer.push(state, action, reward, next_state, done)
                if len(self.replay_buffer) > self.batch_size:
                    pl, vl = self.update()
                    running_ploss += (pl - running_ploss) / (upgrade_steps + 1)
                    running_vloss += (vl - running_vloss) / (upgrade_steps + 1)
                
                state = next_state
                episode_reward += reward
                step += 1
                frame_idx += 1
                
                if frame_idx % self.eval_samples == 0:
                    best_reward = self.evaluation(best_reward, frame_idx)
                
                if done:
                    episode += 1
                    break
            
            rewards.append(episode_reward)
            running_episode_reward += (episode_reward - running_episode_reward) / (episode + 1)
            if len(rewards) < 100:
                running_episode_reward_100 = running_episode_reward
            else:
                last_100 = rewards[-100:]
                running_episode_reward_100 = np.array(last_100).mean()
            self.writer.add_scalar('hp_decay/epsilon', eps, episode)
            self.writer.add_scalar('reward/episode', episode_reward, episode)
            self.writer.add_scalar('reward/running_mean', running_episode_reward, episode)
            self.writer.add_scalar('reward/running_mean_last_100', running_episode_reward_100, episode)
            self.writer.add_scalar('losses/actor_policy', running_ploss, episode)
            self.writer.add_scalar('losses/critic_value', running_vloss, episode)
        # export scalar data to JSON for external processing
        self.writer.export_scalars_to_json("./all_scalars.json")
        self.writer.close()
    
    def update(self):
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        
        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)
        
        policy_loss = self.critic_net(state, self.actor_net(state))
        policy_loss = -policy_loss.mean()
        
        next_action = self.target_policy_net(next_state)
        target_value = self.target_value_net(next_state, next_action.detach())
        expected_value = reward + (1.0 - done) * self.discount * target_value
        
        value = self.critic_net(state, action)
        value_loss = self.critic_loss(value, expected_value.detach())
        
        self.actor_opt.zero_grad()
        policy_loss.backward()
        self.actor_opt.step()
        
        self.critic_opt.zero_grad()
        value_loss.backward()
        self.critic_opt.step()
        
        for target_param, param in zip(self.target_value_net.parameters(), self.critic_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_target_tau) + param.data * self.soft_target_tau
            )
        
        for target_param, param in zip(self.target_policy_net.parameters(), self.actor_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_target_tau) + param.data * self.soft_target_tau
            )
        
        return policy_loss.item(), value_loss.item()

    def evaluation(self, best_reward, frame_idx):
        ts = time.time()
        rewards, steps = self.test()
        print("Test done in %.2f sec, reward %.3f, steps %d" % (
            time.time() - ts, rewards, steps))
        self.writer.add_scalar("test/reward_mean", rewards, frame_idx)
        self.writer.add_scalar("test/steps_mean", steps, frame_idx)
        if best_reward is None or best_reward < rewards:
            if best_reward is not None:
                print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
            fname = self.checkpoint_dir + "best_%+.3f_%d.pth" % (rewards, frame_idx)
            # fname = os.path.join(self.checkpoint_dir, name)
            torch.save(self.actor_net.state_dict(), fname)
            best_reward = rewards
        return best_reward
