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


class AgentDDPG:
    """
    Deep Deterministic Policy Gradient.
    """

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
            batch_size=100,
            n_episode=1000,
            episode_max_len=1000,
            replay_min_size=100,
            replay_max_size=1000000,
            discount=0.99,
            critic_weight_decay=0.,
            critic_update_method='adam',
            critic_lr=1e-3,
            actor_weight_decay=0,
            actor_update_method='adam',
            actor_lr=1e-4,
            eval_samples=10,
            soft_target_tau=0.001,
            n_updates_per_sample=1,
            checkpoint_dir='./checkpoints/',
            tensorboard_dir='./exp/',
            run=0):
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

        self.writer_train = SummaryWriter(tensorboard_dir + 'run_' + str(run) + '/train/')
        self.writer_test = SummaryWriter(tensorboard_dir + 'run_' + str(run) + '/test/')

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
        self.eps = self.eps_start
        self.action_low = torch.tensor(self.env.action_space.low).to(device)
        self.action_high = torch.tensor(self.env.action_space.high).to(device)

        self.critic_net = critic_nn(self.state_dim, self.action_dim).to(device)
        self.actor_net = actor_nn(self.state_dim, self.action_dim).to(device)
        print(self.critic_net)
        print(self.actor_net)

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

        self.episode = 0

    def test(self, count=100):
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
        state = torch.FloatTensor(state).unsqueeze(0).to(device)

        self.eps = self.eps_start - (self.eps_start - self.eps_end) * min(1.0, self.episode / self.eps_decay)
        self.actor_net.eval()
        with torch.no_grad():
            action = self.actor_net(state).cpu().numpy()[0, 0]
        self.actor_net.train()
        # TanH modulation and translation
        # mod = (self.env.action_space.high - self.env.action_space.low) / 2
        # tra = (self.env.action_space.high + self.env.action_space.low) / 2
        # action = mod * action + tra
        if add_noise:
            action = self.noise.get_action(action, self.eps)
        else:
            action = self.noise.get_action(action, 0.0)

        np.clip(action, self.env.action_space.low, self.env.action_space.high)

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
    
    def train(self):
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
                action = self.env.action_space.sample() if self.episode < 10 else self.act(state)

                if self.episode < 10:
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
        # export scalar data to JSON for external processing
        self.writer_train.export_scalars_to_json("./all_scalars.json")
        self.writer_train.close()

    def learn(self, experience, gamma):
        """
        Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
            
        :param experience: tuple of (s, a, r, s', done) tuples
        :param gamma: discount factor
        :return:
        """
        # TODO: check CLAMP or CLIP on everything
        state, action, reward, next_state, done = experience

        # Preparation of the experience
        states = torch.FloatTensor(state).to(device)
        next_states = torch.FloatTensor(next_state).to(device)
        actions = torch.FloatTensor(action).to(device)
        rewards = torch.FloatTensor(reward).unsqueeze(1).to(device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

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
        ts = time.time()
        rewards, steps = self.test()
        print("Test done in %.2f sec, reward %.3f, steps %d" % (
            time.time() - ts, rewards, steps))
        self.writer_test.add_scalar("test/reward_mean", rewards, episode)
        self.writer_test.add_scalar("test/steps_mean", steps, episode)
        if best_reward is None or best_reward < rewards:
            if best_reward is not None:
                print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
            fname_actor = self.checkpoint_dir + "best_actor_%+.3f_%d.pth" % (rewards, episode)
            fname_critic = self.checkpoint_dir + "best_critic_%+.3f_%d.pth" % (rewards, episode)
            
            # fname = os.path.join(self.checkpoint_dir, name)
            torch.save(self.actor_net.state_dict(), fname_actor)
            torch.save(self.critic_net.state_dict(), fname_critic)
            best_reward = rewards
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
