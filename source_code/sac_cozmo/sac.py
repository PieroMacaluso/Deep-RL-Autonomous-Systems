import copy
import datetime
import os
import pickle
import shutil
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
from utils import soft_update, hard_update, copytree


class SAC(object):
    """
    This is the class of SAC Cozmo. It can be used as a starting draft to build your own implementation of SAC on Cozmo.
    The main function to modify as desire is the `train` one.
    """
    
    # TODO: complete documentation of SAC
    def __init__(self, num_inputs, action_space, env, args, folder, logger):
        """
        This is the initialization function of the class. The function receives as input a lot of parameters
        :param num_inputs:
        :type num_inputs:
        :param action_space:
        :type action_space:
        :param env:
        :type env:
        :param args:
        :type args:
        :param folder:
        :type folder:
        :param logger:
        :type logger:
        """
        self.env = env
        self.seed = args.seed
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.learning_rate = args.lr
        
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
        
        # Initialize Critic Network
        self.critic = self.q_network(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.learning_rate)
        # self.scheduler_critic = StepLR(self.critic_optim, 1, gamma=0.99)
        self.critic_target = self.q_network(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)
        
        logger.debug(self.critic)
        
        # Initialize Actor Network
        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.autotune_entropy:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=self.learning_rate)
                # self.scheduler_alpha = StepLR(self.alpha_optim, 1, gamma=0.99)
            
            self.policy = self.gaussian_policy(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=self.learning_rate)
            # self.scheduler_policy = StepLR(self.policy_optim, 1, gamma=0.99)
            logger.debug(self.policy)
        
        else:
            self.alpha = 0
            self.autotune_entropy = False
            self.policy = self.deterministic_policy(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=self.learning_rate)
            # self.scheduler_policy = StepLR(self.policy_optim, 1, gamma=0.99)
        
        self.folder = folder
        self.logger = logger
        
        self.replay_size = args.replay_size
        self.min_replay_size = args.min_replay_size
        self.num_episode = args.num_episode
        self.pics = args.pics
        self.state_buffer_size = args.state_buffer_size
        self.warm_up_episodes = args.warm_up_episodes
        self.batch_size = args.batch_size
        self.updates_per_episode = args.updates_per_episode
        self.eval = args.eval
        self.eval_episode = args.eval_episode
        self.eval_every = args.eval_every
        self.env_name = args.env_name
        self.entropy_backup = None
        self.scale_reward = 1
    
    def select_action(self, state: np.array, eval=False):
        """
        Select the action based on the current state and the current policy network.
        
        :param state: state of the environment
        :type state: np.array
        :param eval: True if we are in the test phase, False otherwise
        :type eval: bool
        :return: Array with the action proposed by the policy network
        :rtype: np.array
        """
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if not eval:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        action = action.detach().cpu().numpy()
        action = action[0]
        assert not np.isnan(action).all()
        # The next 3 lines of code are used to
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
            next_q_value = self.scale_reward * reward_batch + mask_batch * self.gamma * min_qf_next_target
        
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
    
    def train(self, num_run=1, restore=False):
        memory = None
        start_episode = 0
        start_updates = 0
        start_run = 0
        start_total_numsteps = 0
        start_running_episode_reward = 0
        start_running_episode_reward_100 = 0
        start_rewards = []
        start_last_episode_steps = 0
        start_episode_reward = 0
        start_episode_steps = 0
        start_timing = 0
        start_total_timing = 0
        
        # Restore Phase
        if restore:
            # TODO: Not tested deeply yet
            with open(self.folder + "i_episode.pkl", "rb") as pickle_out:
                start_episode = pickle.load(pickle_out)
            with open(self.folder + "i_run.pkl", "rb") as pickle_out:
                start_run = pickle.load(pickle_out)
            with open(self.folder + "memory.pkl", "rb") as pickle_out:
                memory = pickle.load(pickle_out)
            with open(self.folder + "updates.pkl", "rb") as pickle_out:
                start_updates = pickle.load(pickle_out)
            with open(self.folder + "total_numsteps.pkl", "rb") as pickle_out:
                start_total_numsteps = pickle.load(pickle_out)
            with open(self.folder + "running_episode_reward.pkl", "rb") as pickle_out:
                start_running_episode_reward = pickle.load(pickle_out)
            with open(self.folder + "running_episode_reward_100.pkl", "rb") as pickle_out:
                start_running_episode_reward_100 = pickle.load(pickle_out)
            with open(self.folder + "rewards.pkl", "rb") as pickle_out:
                start_rewards = pickle.load(pickle_out)
            with open(self.folder + "last_episode_steps.pkl", "rb") as pickle_out:
                start_last_episode_steps = pickle.load(pickle_out)
            with open(self.folder + "episode_reward.pkl", "rb") as pickle_out:
                start_episode_reward = pickle.load(pickle_out)
            with open(self.folder + "episode_steps.pkl", "rb") as pickle_out:
                start_episode_steps = pickle.load(pickle_out)
            with open(self.folder + "timing.pkl", "rb") as pickle_out:
                start_timing = pickle.load(pickle_out)
            with open(self.folder + "total_timing.pkl", "rb") as pickle_out:
                start_total_timing = pickle.load(pickle_out)
            self.restore_model()
            self.logger.important("Load completed!")

        in_ts = time.time()
        
        # Start of the iteration on runs
        for i_run in range(start_run, num_run):
            
            # Break the loop if the phase "Save'n'Close" is triggered
            if self.env.is_save_and_close():
                break
                
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
            
            # Setup Replay Memory: create new memory if is not the restore case
            if not restore:
                memory = ReplayMemory(self.replay_size)
            # Create a backup memory for Forget-Phase
            backup_memory = copy.deepcopy(memory)
            
            # TRAINING LOOP
            # All these variables must be backed up and restored
            updates = start_updates
            total_numsteps = start_total_numsteps
            running_episode_reward = start_running_episode_reward
            running_episode_reward_100 = start_running_episode_reward_100
            rewards = start_rewards
            i_episode = start_episode
            last_episode_steps = start_last_episode_steps
            episode_reward = start_episode_reward
            episode_steps = start_episode_steps
            timing = start_timing
            total_timing = start_total_timing
            while True:
                
                # Stop the robot
                self.env.stop_all_motors()
                
                # Wait for the human to leave the command
                while self.env.is_human_controlled():
                    pass
                
                # Let's forget (if it is the case)
                if self.env.is_forget_enabled():
                    # print('forget')
                    print(len(memory))
                    self.restore_model()
                    self.env.reset_forget()
                    memory = copy.deepcopy(backup_memory)
                    print(len(memory))
                    # memory.forget_last(last_episode_steps)
                    self.logger.info("Last Episode Forgotten")
                elif i_episode != start_episode:
                    ep_print = i_episode - 1
                    self.print_nets(writer_train, ep_print)
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
                if i_episode % self.eval_every == 0 and self.eval and i_episode != 0:
                    # print('test')
                    self.test_phase(writer_test, i_run, i_episode)
                    # Wait for the human to leave the command
                    while self.env.is_human_controlled():
                        pass
                
                # TODO: HP Checkpoint and check correctness of checkpoint restoring
                if i_episode % 5 == 0 and i_episode != 0 and not restore:
                    self.logger.important("Saving context...")
                    self.logger.info("To restart from here set this flag: --restore " + self.folder)
                    # Save Replay, net weights, hp, i_episode and i_run
                    with open(self.folder + "memory.pkl", "wb") as pickle_out:
                        pickle.dump(memory, pickle_out, pickle.HIGHEST_PROTOCOL)
                    with open(self.folder + "i_episode.pkl", "wb") as pickle_out:
                        pickle.dump(i_episode, pickle_out)
                    with open(self.folder + "i_run.pkl", "wb") as pickle_out:
                        pickle.dump(i_run, pickle_out)
                    with open(self.folder + "updates.pkl", "wb") as pickle_out:
                        pickle.dump(updates, pickle_out)
                    with open(self.folder + "total_numsteps.pkl", "wb") as pickle_out:
                        pickle.dump(total_numsteps, pickle_out)
                    with open(self.folder + "running_episode_reward.pkl", "wb") as pickle_out:
                        pickle.dump(running_episode_reward, pickle_out)
                    with open(self.folder + "running_episode_reward_100.pkl", "wb") as pickle_out:
                        pickle.dump(running_episode_reward_100, pickle_out)
                    with open(self.folder + "rewards.pkl", "wb") as pickle_out:
                        pickle.dump(rewards, pickle_out)
                    with open(self.folder + "last_episode_steps.pkl", "wb") as pickle_out:
                        pickle.dump(last_episode_steps, pickle_out)
                    with open(self.folder + "episode_reward.pkl", "wb") as pickle_out:
                        pickle.dump(episode_reward, pickle_out)
                    with open(self.folder + "episode_steps.pkl", "wb") as pickle_out:
                        pickle.dump(episode_steps, pickle_out)
                    with open(self.folder + "timing.pkl", "wb") as pickle_out:
                        pickle.dump(timing, pickle_out)
                    with open(self.folder + "total_timing.pkl", "wb") as pickle_out:
                        pickle.dump(total_timing, pickle_out)
                    self.backup_model()
                    if os.path.exists(self.folder[:-1] + "_bak" + self.folder[-1:]):
                        shutil.rmtree(self.folder[:-1] + "_bak" + self.folder[-1:])
                    print(self.folder[:-1] + "_bak" + self.folder[-1:])
                    shutil.copytree(self.folder, self.folder[:-1] + "_bak" + self.folder[-1:])
                    self.logger.important("Save completed!")
                
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
                updates_episode = 0
                
                # Start of the episode
                while not done:
                    if self.pics:
                        writer_train.add_image('episode_{}'
                                               .format(str(i_episode)), state_buffer.get_tensor(), episode_steps)
                    
                    if i_episode < self.warm_up_episodes or len(memory) < self.min_replay_size:
                        # Warm_up phase -> Completely random choice of an action
                        action = self.env.action_space.sample()
                    else:
                        # Training phase -> Action sampled from policy
                        action = self.select_action(state)
                    
                    assert action.shape == self.env.action_space.shape
                    assert action is not None
                    # TODO: problem with histograms!
                    # print(action)
                    writer_train.add_histogram('action_speed/episode_{}'
                                               .format(str(i_episode)), torch.tensor(action[0]), episode_steps)
                    writer_train.add_histogram('action_turn/episode_{}'
                                               .format(str(i_episode)), torch.tensor(action[1]), episode_steps)
                    
                    # Make the action
                    next_state, reward, done, info = self.env.step(action)
                    
                    # Save the step
                    if self.pics:
                        state_buffer.push(next_state)
                        next_state = state_buffer.get_state()
                    episode_steps += 1
                    total_numsteps += 1
                    episode_reward += reward
                    mask = 1 if done else float(not done)
                    
                    # Push the transition in the memory only if n steps is greater than 5
                    # print('push')
                    if episode_steps > 5:
                        memory.push(state, action, reward, next_state, mask)
                    state = next_state
                if self.env.is_forget_enabled():
                    continue
                print("Memory {}/{}".format(len(memory), self.replay_size))
                if len(memory) > self.min_replay_size and i_episode > self.warm_up_episodes:
                    updates = self.learning_phase(self.updates_per_episode, memory, updates, writer_learn)
                    mem_size_last_learn = total_numsteps
                    updates_episode += self.updates_per_episode
                # self.logger.info("#TotalUpdates={})"
                #                  .format(updates))
                # self.scheduler_alpha.step()
                # self.scheduler_critic.step()
                # self.scheduler_policy.step()
                # print("{} {} {}"
                #       .format(self.scheduler_policy.get_lr(), self.scheduler_critic.get_lr(),
                #               self.scheduler_alpha.get_lr()))
                last_episode_steps = episode_steps
                i_episode += 1
                timing = time.time() - ts
                total_timing = time.time() - in_ts
                start_episode = 0
                # Disable restore phase after the restored run
                restore = False
    
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
        if self.autotune_entropy:
            # entropy_t_path = model_f + f"sac_entropy_t"
            log_alpha_path = model_f + f"sac_log_alpha"
            # torch.save(self.target_entropy, entropy_t_path)
            torch.save(self.log_alpha, log_alpha_path)
    
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
        if self.autotune_entropy:
            # entropy_t_path = model_f + f"sac_entropy_t"
            log_alpha_path = model_f + f"sac_log_alpha"
            # self.target_entropy = torch.load(entropy_t_path)
            self.log_alpha = torch.load(log_alpha_path)
            self.alpha_optim = Adam([self.log_alpha], lr=self.learning_rate)
    
    def test_phase(self, writer_test, i_run, i_episode):
        n_tests = 0
        ts = time.time()
        total_reward = 0
        for _ in range(self.eval_episode):
            while self.env.is_human_controlled():
                pass
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
            writer_learn.add_scalar('entropy_temperature/learning_rate', torch.tensor(self.learning_rate),
                                    updates)
            
            updates += 1
        # print(updates)
        self.logger.info("Update (up. {})took {}s"
                         .format(updates_per_episode,
                                 round(time.time() - time_update, 2)))
        return updates
    
    def print_nets(self, writer_train: SummaryWriter, ep_print: int):
        for k, v in self.policy.state_dict().items():
            # print(k)
            if (k.endswith('bias') or k.endswith('weight')) and (k.startswith('conv') or k.startswith('conv')):
                writer_train.add_histogram('policy/' + k, v, global_step=ep_print)
        for k, v in self.critic.state_dict().items():
            if (k.endswith('bias') or k.endswith('weight')) and (k.startswith('conv') or k.startswith('conv')):
                writer_train.add_histogram('critic/' + k, v, global_step=ep_print)
        for k, v in self.critic_target.state_dict().items():
            if (k.endswith('bias') or k.endswith('weight')) and (k.startswith('conv') or k.startswith('conv')):
                writer_train.add_histogram('critic_target/' + k, v, global_step=ep_print)
        
        pass
