import os
import time

import gym
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import optim, nn

from CNN import CriticCNN, ActorCNN
from NN import CriticNN, ActorNN
from OUNoise import OUNoise
from ReplayBuffer import ReplayBuffer
from StateBuffer import StateBuffer


class DDPG:
    """
    Deep Deterministic Policy Gradient.
    """
    
    def __init__(self,
                 env: gym.Env,
                 test_env: gym.Env,
                 exp_strategy: OUNoise = OUNoise,
                 eps_start=0.9,
                 eps_end=0.2,
                 eps_decay=1000,
                 batch_size=64,
                 n_episode=1000,
                 episode_max_len=1000,
                 replay_min_size=80,
                 replay_max_size=1000000,
                 discount=0.99,
                 critic_weight_decay=0.,
                 critic_update_method='adam',
                 critic_lr=1e-3,
                 actor_weight_decay=0,
                 actor_update_method='adam',
                 actor_lr=1e-4,
                 size_state_buffer=3,
                 eval_samples=10000,
                 soft_target_tau=0.001,
                 n_updates_per_sample=1,
                 checkpoint_dir='./checkpoints/',
                 timestamp='0000',
                 run=0,
                 pics=True):
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
        self.warm_up = 10
        self.state_batch_size = 2
        self.id = 'DDPG'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.env = env
        self.test_env = test_env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.size_state_buffer = size_state_buffer
        
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
        self.eps = eps_start
        
        if pics:
            # CNN
            self.train = self.train_cnn
            self.test = self.test_cnn
            self.h = env.observation_space.shape[1]
            self.w = env.observation_space.shape[2]
            self.critic_net = CriticCNN(self.size_state_buffer, self.state_batch_size, self.action_dim, self.h, self.w).to(
                self.device)
            self.actor_net = ActorCNN(self.size_state_buffer, self.state_batch_size, self.action_dim, self.h, self.w).to(self.device)
            self.target_value_net = CriticCNN(self.size_state_buffer, self.state_batch_size, self.action_dim, self.h, self.w).to(
                self.device)
            self.target_policy_net = ActorCNN(self.size_state_buffer, self.state_batch_size, self.action_dim, self.h, self.w).to(
                self.device)
        else:
            # NN
            self.train = self.train_nn
            self.test = self.test_nn
            self.critic_net = CriticNN(self.state_dim, self.action_dim).to(self.device)
            self.actor_net = ActorNN(self.state_dim, self.action_dim).to(self.device)
            self.target_value_net = CriticNN(self.state_dim, self.action_dim).to(self.device)
            self.target_policy_net = ActorNN(self.state_dim, self.action_dim).to(self.device)
        
        self.folder = '{}_{}_{}_{}/'.format(timestamp, self.id,
                                            'CNN' if pics else 'NN', env.spec.id)
        
        self.writer_train = SummaryWriter(log_dir='runs/' + self.folder + 'run_' + str(run) + '/train/')
        self.writer_test = SummaryWriter(log_dir='runs/' + self.folder + 'run_' + str(run) + '/test/')
        
        # Useful for plotting graph in tensorboard
        dummy_state = StateBuffer(self.state_batch_size, env.reset()).get_state()
        dummy_action = env.action_space.sample()
        dummy_state = torch.FloatTensor(dummy_state).to(self.device).unsqueeze(0)
        dummy_action = torch.FloatTensor(dummy_action).to(self.device)
        with SummaryWriter(log_dir='runs/' + self.folder + 'run_' + str(run) + '/actor/', comment='actor') as w:
            w.add_graph(self.actor_net, dummy_state, verbose=True)
        with SummaryWriter(log_dir='runs/' + self.folder + 'run_' + str(run) + '/critic/', comment='critic') as w:
            w.add_graph(self.critic_net, (dummy_state, dummy_action,), verbose=True)
        
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
        
        self.checkpoint_dir = './checkpoint/' + self.folder + 'run_' + str(run)
        self.episode = 0
    
    def test_cnn(self, count=1):
        """
        Testing the actor on a random test set without using OUNoise.

        :param count: number of episode to test.
        :return: averages of rewards and steps.
        """
        rewards = 0.0
        steps = 0
        for _ in range(count):
            t = 0
            done = False
            old = self.test_env.reset()
            state_buffer = StateBuffer(self.state_batch_size, old)
            while t < self.episode_max_len:
                state = state_buffer.get_state()
                action = self.act(state, add_noise=False)
                next_state, reward, done, _ = self.test_env.step(action)
                # self.test_env.render()
                state_buffer.push(next_state)
                rewards += reward
                t += 1
                if done:
                    break
            if not done:
                t += 1
            steps += t
        print(
            "Test Episode ({} ep.): reward: {}, steps: {}".format(count, round(rewards / count, 2),
                                                                  round(steps / count)))
        return rewards / count, steps / count
    
    def test_nn(self, count=1):
        """
        Testing the actor on a random test set without using OUNoise.

        :param count: number of episode to test.
        :return: averages of rewards and steps.
        """
        rewards = 0.0
        steps = 0
        for _ in range(count):
            t = 0
            state = self.test_env.reset()
            done = False
            while t < self.episode_max_len:
                action = self.act(state, add_noise=False)
                next_state, reward, done, _ = self.test_env.step(action)
                # self.test_env.render()
                state = next_state
                rewards += reward
                t += 1
                if done:
                    break
            if not done:
                t += 1
            steps += t
        print(
            "Test Episode ({} ep.): reward: {}, steps: {}".format(count, round(rewards / count, 2),
                                                                  round(steps / count)))
        return rewards / count, steps / count
    
    def reset(self):
        """
        Resets the environment and the Noise.

        :return: The initial state of the environment.
        """
        self.noise.reset()
        return self.env.reset()
    
    def act(self, state, add_noise=True):
        """
        The ACT part of the code.
        In this part the state is used to get the next action following the policy given by the actor net. Finally we add
        noise to the resulting action and clip the action-values in the proper range.

        :param state: Environment State.
        :param add_noise: it is True if is needed to add noise, False otherwise.
        :return: The Action to be performed in the STEP part.
        """
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        
        self.eps = self.eps_start - (self.eps_start - self.eps_end) * min(1.0, self.episode / self.eps_decay)
        self.actor_net.eval()
        with torch.no_grad():
            action = self.actor_net(state).cpu().numpy()
            action = action[0]
        
        self.actor_net.train()
        
        # i = np.argmin(action[1:3])
        # action[1+i] = 0
        if add_noise:
            action = self.noise.get_action(action, self.eps)
        else:
            action = self.noise.get_action(action, 0.0)
        
        action = np.clip(action, -1, 1)
        # assert -1 <= action.all() <= 1, "Action OOR"
        
        return action
    
    def step(self, action):
        """
        Given the action, perform the STEP part.

        :param action: action to be performed.
        :return: next_state, reward and done flag.
        """
        next_state, reward, done, _ = self.env.step(action)
        # self.env.render()
        
        return next_state, reward, done
    
    def train_cnn(self):
        """
        TRAIN part of the code. It is the main Agent.
        It contains the main loop and the coordination among all the parts of the code.

        :return: Nothing
        """
        self.episode = running_episode_reward_100 = running_episode_reward = frame_idx = 0
        rewards = []
        best_reward = None
        while self.episode < self.n_episode:
            ts = time.time()
            episode_reward = upgrade_steps = running_ploss = running_vloss = step = 0
            done = False
            old = self.reset()
            state_buffer = StateBuffer(self.state_batch_size, old)
            self.noise.reset()
            while step < self.episode_max_len:
                state = state_buffer.get_state()
                self.writer_train.add_images('episode_{}'.format(str(self.episode)), state_buffer.get_tensor(), step)
                # TODO: CNN
                action = self.env.action_space.sample() if self.episode < self.warm_up else self.act(state)
                self.writer_train.add_histogram('act_episode_{}'.format(str(self.episode)), action, step)
                if self.episode < self.warm_up:
                    next_state, reward, done, _ = self.env.step(action)
                else:
                    next_state, reward, done = self.step(action)
                state_buffer.push(next_state)
                self.replay_buffer.push(state, action, reward, state_buffer.get_state(), done)
                if frame_idx > self.replay_min_size:
                    # pl, vl = self.update()
                    experience = self.replay_buffer.sample(self.batch_size)
                    pl, vl = self.learn(experience, self.discount)
                    
                    running_ploss += (pl - running_ploss) / (upgrade_steps + 1)
                    running_vloss += (vl - running_vloss) / (upgrade_steps + 1)
                
                episode_reward += reward
                step += 1
                frame_idx += 1
                
                if done:
                    self.episode += 1
                    break
            if not done:
                self.episode += 1
            
            if self.episode % self.eval_samples == 0:
                best_reward = self.evaluation(best_reward, self.episode)
            
            rewards.append(episode_reward)
            running_episode_reward += (episode_reward - running_episode_reward) / self.episode
            if len(rewards) < 100:
                running_episode_reward_100 = running_episode_reward
            else:
                last_100 = rewards[-100:]
                running_episode_reward_100 = np.array(last_100).mean()
            self.writer_train.add_scalar('hp/epsilon', self.eps, self.episode)
            self.writer_train.add_scalar('losses/actor_policy', running_ploss, self.episode)
            self.writer_train.add_scalar('losses/critic_value', running_vloss, self.episode)
            self.writer_train.add_scalar('reward/episode', episode_reward, self.episode)
            self.writer_train.add_scalar('reward/running_mean', running_episode_reward, self.episode)
            self.writer_train.add_scalar('reward/running_mean_last_100', running_episode_reward_100, self.episode)
            print("ep {}/{}, t {}, r_t {}, 100_mean {}, time_spent {}s ".format(self.episode, self.n_episode,
                                                                               step,
                                                                               round(episode_reward, 2),
                                                                               round(running_episode_reward_100, 2),
                                                                               round(time.time() - ts, 2)))
        self.writer_train.close()
    
    def train_nn(self):
        """
        TRAIN part of the code. It is the main Agent.
        It contains the main loop and the coordination among all the parts of the code.

        :return: Nothing
        """
        self.episode = running_episode_reward_100 = running_episode_reward = frame_idx = 0
        best_reward = None
        rewards = []
        
        while self.episode < self.n_episode:
            episode_reward = upgrade_steps = running_ploss = running_vloss = step = 0
            state = self.reset()
            done = False
            while step < self.episode_max_len:
                action = self.env.action_space.sample() if self.episode < self.warm_up else self.act(state)
                
                if self.episode < self.warm_up:
                    next_state, reward, done, _ = self.env.step(action)
                else:
                    next_state, reward, done = self.step(action)
                self.replay_buffer.push(state, action, reward, next_state, done)
                
                if frame_idx > self.replay_min_size:
                    experience = self.replay_buffer.sample(self.batch_size)
                    pl, vl = self.learn(experience, self.discount)
                    
                    running_ploss += (pl - running_ploss) / (upgrade_steps + 1)
                    running_vloss += (vl - running_vloss) / (upgrade_steps + 1)
                
                state = next_state
                episode_reward += reward
                step += 1
                frame_idx += 1
                
                if done:
                    self.episode += 1
                    break
            if not done:
                self.episode += 1
            
            if self.episode % self.eval_samples == 0:
                best_reward = self.evaluation(best_reward, self.episode)
            
            rewards.append(episode_reward)
            running_episode_reward += (episode_reward - running_episode_reward) / self.episode
            if len(rewards) < 100:
                running_episode_reward_100 = running_episode_reward
            else:
                last_100 = rewards[-100:]
                running_episode_reward_100 = np.array(last_100).mean()
            self.writer_train.add_scalar('hp/epsilon', self.eps, self.episode)
            self.writer_train.add_scalar('losses/actor_policy', running_ploss, self.episode)
            self.writer_train.add_scalar('losses/critic_value', running_vloss, self.episode)
            self.writer_train.add_scalar('reward/episode', episode_reward, self.episode)
            self.writer_train.add_scalar('reward/running_mean', running_episode_reward, self.episode)
            self.writer_train.add_scalar('reward/running_mean_last_100', running_episode_reward_100, self.episode)
            print("ep {}/{}, \t t {}, \t  r_t {}, \t 100_mean {}".format(self.episode, self.n_episode,
                                                                         step,
                                                                         round(episode_reward, 2),
                                                                         round(running_episode_reward_100, 2)))
        self.writer_train.close()
    
    def learn(self, experience, gamma):
        state, action, reward, next_state, done = experience
        
        # Preparation of the experience
        states = torch.FloatTensor(state).to(self.device)
        next_states = torch.FloatTensor(next_state).to(self.device)
        actions = torch.FloatTensor(action).to(self.device)
        rewards = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)
        
        # UPDATE CRITIC #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.target_policy_net(next_states)
        q_targets_next = self.target_value_net(next_states, actions_next.detach())
        # Compute Q targets for current states (y_i)
        q_targets = rewards + (gamma * q_targets_next * (1.0 - done))
        # Compute critic loss
        q_expected = self.critic_net(states, actions)
        critic_loss = self.critic_loss(q_expected, q_targets)
        # Minimize the loss
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        
        # UPDATE ACTOR #
        # Compute actor loss
        actions_pred = self.actor_net(states)
        actor_loss = -self.critic_net(states, actions_pred).mean()
        # Maximize the expected return
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        
        # UPDATE TARGET NETWORK #
        self.soft_update(self.critic_net, self.target_value_net, self.soft_target_tau)
        self.soft_update(self.actor_net, self.target_policy_net, self.soft_target_tau)
        
        return actor_loss.item(), critic_loss.item()
    
    def evaluation(self, best_reward, episode):
        """
        Evaluation of the model currently discovered.

        :param best_reward: The best reward found till now.
        :param episode: counter of the frame used till now.
        :return: the new best reward
        """
        print("----------------------------------------")
        ts = time.time()
        rewards, steps = self.test()
        print("Test done in %.2f sec, reward %.3f, steps %d" % (
            time.time() - ts, rewards, steps))
        self.writer_test.add_scalar("test/reward_mean", rewards, episode)
        self.writer_test.add_scalar("test/steps_mean", steps, episode)
        if best_reward is None or best_reward < rewards:
            if best_reward is not None:
                print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
            fname = self.checkpoint_dir + "best_%+.3f_%d.pth" % (rewards, episode)
            # fname = os.path.join(self.checkpoint_dir, name)
            self.save_model(self.env.spec.id)
            # torch.save(self.actor_net.state_dict(), fname)
            best_reward = rewards
        print("----------------------------------------")
        
        return best_reward
    
    def soft_update(self, local_net, target_net, tau):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        :param local_net: PyTorch model (weights will be copied from)
        :param target_net: PyTorch model (weights will be copied to)
        :param tau: interpolation parameter
        :return: nothing.
        """
        for target_param, local_param in zip(target_net.parameters(), local_net.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    
    # Save model parameters
    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')
        
        if actor_path is None:
            actor_path = "models/{}_actor_{}_{}".format(self.id, env_name, suffix)
        if critic_path is None:
            critic_path = "models/{}_critic_{}_{}".format(self.id, env_name, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.actor_net.state_dict(), actor_path)
        torch.save(self.critic_net.state_dict(), critic_path)
    
    # Load model parameters
    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.actor_net.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic_net.load_state_dict(torch.load(critic_path))
