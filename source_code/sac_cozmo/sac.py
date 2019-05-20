import copy
import datetime
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.optim import Adam

from model import GaussianPolicyCNN, QNetworkCNN, DeterministicPolicyCNN
from model import GaussianPolicyNN, QNetworkNN, DeterministicPolicyNN
from replay_memory import ReplayMemory
from state_buffer import StateBuffer
from utils import soft_update, hard_update


class SAC(object):
    def __init__(self, num_inputs, action_space, env, args, folder, logger):
        
        self.env = env
        self.seed = args.seed
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        
        self.policy_type = args.policy
        self.target_update = args.target_update
        self.autotune_entropy = args.autotune_entropy
        
        self.pics = args.pics
        if self.pics:
            self.q_network = QNetworkCNN
            self.gaussian_policy = GaussianPolicyCNN
            self.deterministic_policy = DeterministicPolicyCNN
        else:
            self.q_network = QNetworkNN
            self.gaussian_policy = GaussianPolicyNN
            self.deterministic_policy = DeterministicPolicyNN
        
        self.critic = self.q_network(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)
        logger.debug(self.critic)
        
        self.critic_target = self.q_network(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)
        
        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.autotune_entropy:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)
            
            self.policy = self.gaussian_policy(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
            logger.debug(self.policy)
        
        else:
            self.alpha = 0
            self.autotune_entropy = False
            self.policy = self.deterministic_policy(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
        
        self.folder = folder
        self.logger = logger
        
        self.replay_size = args.replay_size
        self.num_episode = args.num_episode
        self.pics = args.pics
        self.state_buffer_size = args.state_buffer_size
        self.warm_up_steps = args.warm_up_steps
        self.batch_size = args.batch_size
        self.updates_per_episode = args.updates_per_episode
        self.eval_episode = args.eval_episode
        self.env_name = args.env_name
        self.entropy_backup = None
    
    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if not eval:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        action = action.detach().cpu().numpy()
        action = action[0]
        mod = (self.env.action_space.high - self.env.action_space.low) / 2
        tra = (self.env.action_space.high + self.env.action_space.low) / 2
        action = action * mod + tra
        return action
    
    def update_parameters(self, memory, batch_size, updates):
        """

        :param memory:
        :param batch_size:
        :param updates:
        :return:
        """
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)
        
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        
        # V(st+1) = 𝔼(at~D)
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * min_qf_next_target
        
        # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        
        pi, log_pi, _ = self.policy.sample(state_batch)
        
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        
        # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
        
        self.critic_optim.zero_grad()
        qf1_loss.backward()
        self.critic_optim.step()
        
        self.critic_optim.zero_grad()
        qf2_loss.backward()
        self.critic_optim.step()
        
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        
        if self.autotune_entropy:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            
            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs
        
        if updates % self.target_update == 0:
            soft_update(self.critic_target, self.critic, self.tau)
        
        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()
    
    def train(self, num_run=1):
        in_ts = time.time()
        for i_run in range(num_run):
            self.logger.important(f"START TRAINING RUN {i_run}")
            
            # Set Seed for repeatability
            torch.manual_seed(self.seed + i_run)
            np.random.seed(self.seed + i_run)
            self.env.seed(self.seed + i_run)
            self.env.action_space.np_random.seed(self.seed + i_run)
            
            # Setup TensorboardX
            writer_train = SummaryWriter(log_dir=self.folder + 'run_' + str(i_run) + '/train')
            writer_learn = SummaryWriter(log_dir=self.folder + 'run_' + str(i_run) + '/learn')
            writer_test = SummaryWriter(log_dir=self.folder + 'run_' + str(i_run) + '/test')
            
            # Setup Replay Memory
            memory = ReplayMemory(self.replay_size)
            backup_memory = copy.deepcopy(memory)
            # TRAINING LOOP
            total_numsteps = updates = running_episode_reward = running_episode_reward_100 = 0
            rewards = []
            i_episode = 0
            last_episode_steps = 0
            episode_reward = episode_steps = timing = total_timing = 0
            while True:
                
                # Stop the robot
                self.env.stop_all_motors()
                
                # Wait for the human to leave the command
                while self.env.is_human_controlled():
                    continue
                
                # Let's forget (if it is the case)
                if self.env.is_forget_enabled():
                    print(len(memory))
                    self.restore_model()
                    self.env.reset_forget()
                    memory = copy.deepcopy(backup_memory)
                    print(len(memory))
                    # memory.forget_last(last_episode_steps)
                    self.logger.info("Last Episode Forgotten")
                elif i_episode != 0:
                    ep_print = i_episode - 1
                    rewards.append(episode_reward)
                    running_episode_reward += (episode_reward - running_episode_reward) / (ep_print + 1)
                    if len(rewards) < 100:
                        running_episode_reward_100 = running_episode_reward
                    else:
                        last_100 = rewards[-100:]
                        running_episode_reward_100 = np.array(last_100).mean()
                    
                    writer_train.add_scalar('reward/train', episode_reward, ep_print)
                    writer_train.add_scalar('reward/steps', last_episode_steps, ep_print)
                    writer_train.add_scalar('reward/running_mean', running_episode_reward, ep_print)
                    writer_train.add_scalar('reward/running_mean_last_100', running_episode_reward_100, ep_print)
                    self.logger.info("Ep. {}/{}, t {}, r_t {}, 100_mean {}, time_spent {}s | {}s "
                                     .format(ep_print, self.num_episode, episode_steps, round(episode_reward, 2),
                                             round(running_episode_reward_100, 2), round(timing, 2),
                                             str(datetime.timedelta(seconds=total_timing))))
                
                # Let's test (if it is the case)
                while self.env.is_test_phase():
                    self.test_phase(writer_test, i_run, i_episode)
                    # Wait for the human to leave the command
                    while self.env.is_human_controlled():
                        continue
                
                # Limit of episode/run reached. Let's start a new RUN
                if i_episode > self.num_episode:
                    break
                
                # Backup NNs and memory (useful in case of Forget Phase)
                self.backup_model()
                backup_memory = copy.deepcopy(memory)
                
                # Setup the episode
                self.logger.important(f"START EPISODE {i_episode}")
                ts = time.time()
                episode_reward = episode_steps = 0
                done = False
                info = {'undo': False}
                state = self.env.reset()
                state_buffer = None
                
                # If you use CNNs, the use of StateBuffer is enabled (see doc).
                if self.pics:
                    state_buffer = StateBuffer(self.state_buffer_size, state)
                    state = state_buffer.get_state()
                
                # Start of the episode
                while not done:
                    if self.pics:
                        writer_train.add_image('episode_{}'
                                               .format(str(i_episode)), state_buffer.get_tensor(), episode_steps)
                    
                    if episode_steps < self.warm_up_steps:
                        # Warm_up phase -> Completely random choice of an action
                        action = self.env.action_space.sample()
                        print("random " + action)
                    else:
                        # Training phase -> Action sampled from policy
                        action = self.select_action(state)
                        print("net " + action)
                    if len(memory) > self.batch_size:  # and not self.env.is_forget_enabled():
                        updates = self.learning_phase(1, memory, updates, writer_learn)
                    
                    # Make the action and get the new context
                    next_state, reward, done, info = self.env.step(action)
                    if self.pics:
                        state_buffer.push(next_state)
                        next_state = state_buffer.get_state()
                    episode_steps += 1
                    total_numsteps += 1
                    episode_reward += reward
                    mask = 1 if done else float(not done)
                    
                    # Push the transition in the memory
                    memory.push(state, action, reward, next_state, mask)
                    state = next_state
                # while len(memory) > self.batch_size and updates < (i_episode+1)*200:
                #     updates = self.learning_phase(1, memory, updates, writer_learn)
                # self.logger.info("#TotalUpdates={})"
                #                  .format(updates))
                last_episode_steps = episode_steps
                i_episode += 1
                timing = time.time() - ts
                total_timing = time.time() - in_ts
            
            # self.env.close()
    
    def do_one_test(self):
        old = self.env.reset()
        state_buffer = StateBuffer(self.state_buffer_size, old)
        episode_reward = 0
        done = False
        while not done:
            state = state_buffer.get_state()
            action = self.select_action(state, eval=True)
            
            next_state, reward, done, _ = self.env.step(action)
            episode_reward += reward
            
            state_buffer.push(next_state)
        return episode_reward
    
    # Save model parameters
    def save_model(self, env_name, folder, i_episode, suffix=""):
        model_f = folder + 'models/' + f"episode_{i_episode}/"
        if not os.path.exists(model_f):
            os.makedirs(model_f)
        
        actor_path = model_f + f"sac_actor_{env_name}_episode{i_episode}"
        critic_path = model_f + f"sac_critic_{env_name}_episode{i_episode}"
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
    
    # Load model parameters
    def load_model(self, actor_path, critic_path):
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))
    
    # Backup model parameters
    def backup_model(self):
        model_f = self.folder + 'backup/'
        if not os.path.exists(model_f):
            os.makedirs(model_f)
        
        actor_path = model_f + f"sac_actor"
        critic_path = model_f + f"sac_critic"
        critic_t_path = model_f + f"sac_critic_t"
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        torch.save(self.critic_target.state_dict(), critic_t_path)
        if self.autotune_entropy and self.entropy_backup is None:
            self.entropy_backup = self.target_entropy
    
    # Restore model parameters
    def restore_model(self):
        model_f = self.folder + 'backup/'
        actor_path = model_f + f"sac_actor"
        critic_path = model_f + f"sac_critic"
        critic_t_path = model_f + f"sac_critic_t"
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))
        if critic_t_path is not None:
            self.critic_target.load_state_dict(torch.load(critic_t_path))
        if self.autotune_entropy and self.entropy_backup is not None:
            self.target_entropy = self.entropy_backup
            self.entropy_backup = None
    
    def test_phase(self, writer_test, i_run, i_episode):
        n_tests = 0
        ts = time.time()
        total_reward = 0
        while self.env.is_test_phase():
            while self.env.is_human_controlled():
                continue
            if self.env.is_forget_enabled():
                n_tests -= 1
                self.logger.info("Last Test Episode Forgotten")
            episode_reward = self.do_one_test()
            total_reward += episode_reward
            n_tests += 1
        
        writer_test.add_scalar('reward/test', total_reward / n_tests, i_episode)
        
        self.logger.info("----------------------------------------")
        self.logger.info("Test {} ep.: {}, mean_r: {}, time_spent {}s"
                         .format(self.eval_episode,
                                 i_episode,
                                 round(total_reward / self.eval_episode, 2),
                                 round(time.time() - ts, 2)))
        self.save_model(self.env_name, self.folder + f"run_{i_run}/", i_episode)
        self.logger.info('Saving models...')
        self.logger.info("----------------------------------------")
    
    def learning_phase(self, updates_per_episode, memory, updates, writer_learn):
        time_update = time.time()
        # Let's update our parameters, this is the main part of learning
        for i in range(updates_per_episode):
            # Update parameters of all the networks
            critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = self.update_parameters(memory,
                                                                                                self.batch_size,
                                                                                                updates)
            writer_learn.add_scalar('loss/critic_1', critic_1_loss, updates)
            writer_learn.add_scalar('loss/critic_2', critic_2_loss, updates)
            writer_learn.add_scalar('loss/policy', policy_loss, updates)
            writer_learn.add_scalar('loss/entropy_loss', ent_loss, updates)
            writer_learn.add_scalar('entropy_temperature/alpha', alpha, updates)
            updates += 1
        # print(updates)
        # self.logger.info("Update (up. {})took {}s"
        #                  .format(updates_per_episode,
        #                          round(time.time() - time_update, 2)))
        return updates
