import copy
import datetime
import json
import os
import pickle
import shutil
import time

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import optim, nn

from CNN import CriticCNN, ActorCNN
from NN import CriticNN, ActorNN
from OUNoise import OUNoise
from replay_memory import ReplayMemory
from state_buffer import StateBuffer
from utils import soft_update, hard_update


class DDPG(object):
    """
    This is the class of SAC Cozmo. It can be used as a starting draft to build your own implementation of SAC on Cozmo.
    The main function to modify as desire is the `train` one.
    """
    
    # TODO: complete documentation of DDPG
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
        self.learning_rate = args.lr
        
        self.target_update = args.target_update
        self.eps_start = args.eps_start
        self.eps_end = args.eps_end
        self.eps_decay = args.eps_decay
        self.eps = args.eps_start
        self.noise = OUNoise(env.action_space, mu=args.noise[0], sigma=args.noise[1], theta=args.noise[2])
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
        
        self.pics = args.pics
        if self.pics:
            # CNN
            self.h = args.img_h
            self.w = args.img_w
            self.critic_net = CriticCNN(num_inputs, action_space.shape[0], self.h,
                                        self.w).to(
                self.device)
            self.actor_net = ActorCNN(num_inputs, action_space.shape[0], self.h,
                                      self.w).to(self.device)
            self.target_value_net = CriticCNN(num_inputs, action_space.shape[0], self.h,
                                              self.w).to(
                self.device)
            self.target_policy_net = ActorCNN(num_inputs, action_space.shape[0], self.h,
                                              self.w).to(
                self.device)
        else:
            # NN
            self.critic_net = CriticNN(action_space, num_inputs).to(self.device)
            self.actor_net = ActorNN(action_space, num_inputs).to(self.device)
            self.target_value_net = CriticNN(action_space, num_inputs).to(self.device)
            self.target_policy_net = ActorNN(action_space, num_inputs).to(self.device)
        
        hard_update(self.target_value_net, self.critic_net)
        hard_update(self.target_policy_net, self.actor_net)
        
        self.critic_opt = optim.Adam(self.critic_net.parameters(), lr=self.learning_rate)
        self.actor_opt = optim.Adam(self.actor_net.parameters(), lr=self.learning_rate)
        self.critic_loss = nn.MSELoss()
        
        logger.debug(self.critic_net)
        logger.debug(self.actor_net)
        
        self.folder = folder
        self.logger = logger
    
    def select_action(self, state: np.array, i_episode, eval=False):
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
        action = self.actor_net.sample(state)
        action = action.detach().cpu().numpy()
        action = action[0]

        # The next 3 lines of code are used to
        mod = (self.env.action_space.high - self.env.action_space.low) / 2
        tra = (self.env.action_space.high + self.env.action_space.low) / 2
        action = action * mod + tra
        if not eval:
            self.eps = self.eps_start - (self.eps_start - self.eps_end) * min(1.0, i_episode / self.eps_decay)
            action = self.noise.get_action(action, self.eps)
        else:
            action = self.noise.get_action(action, 0.0)
        assert not np.isnan(action).all()
        return action
    
    def update_parameters(self, memory, batch_size, updates):
        """

        :param memory:
        :param batch_size:
        :param updates:
        :return:
        """
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)
        
        states = torch.FloatTensor(state_batch).to(self.device)
        next_states = torch.FloatTensor(next_state_batch).to(self.device)
        actions = torch.FloatTensor(action_batch).to(self.device)
        rewards = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        done = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        
        # UPDATE CRITIC #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.target_policy_net(next_states)
        q_targets_next = self.target_value_net(next_states, actions_next.detach())
        # Compute Q targets for current states (y_i)
        q_targets = rewards + (self.gamma * q_targets_next * (1.0 - done))
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
        if updates % self.target_update == 0:
            soft_update(self.critic_net, self.target_value_net, self.tau)
            soft_update(self.actor_net, self.target_policy_net, self.tau)
        
        return actor_loss.item(), critic_loss.item()
    
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
            with open(self.folder + "memory.pkl", "rb") as pickle_out:
                memory = ReplayMemory(self.replay_size, self.seed)
                memory.load(pickle_out)
            with open(self.folder + "context.json", "r+") as pickle_out:
                (start_episode, start_run, start_updates, start_total_numsteps, start_running_episode_reward,
                 start_running_episode_reward_100, start_last_episode_steps, start_episode_reward, start_episode_steps,
                 start_timing, start_total_timing) = json.load(pickle_out)
            with open(self.folder + "rewards.pkl", "rb") as pickle_out:
                start_rewards = pickle.load(pickle_out)
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
            
            # Setup Replay Memory
            if not restore:
                memory = ReplayMemory(self.replay_size, self.seed)
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
            updates_episode = 0
            episode_images = list()
            
            while True:
                
                # Stop the robot
                self.env.stop_all_motors()
                
                # Wait for the human to leave the command
                while self.env.is_human_controlled():
                    pass

                # Let's forget (if it is the case)
                if self.env.is_forget_enabled():
                    # print('forget')
                    i_episode -= 1
                    print(len(memory))
                    # Restore Nets
                    self.restore_model()
                    self.env.reset_forget()
                    # Restore Memory
                    memory = copy.deepcopy(backup_memory)
                    print(len(memory))
                    # memory.forget_last(last_episode_steps)
                    self.logger.info("Last Episode Forgotten")
                elif i_episode != start_episode:
                    # LEARNING AND PRINTING PHASE
                    ep_print = i_episode - 1
                    last_episode_steps = episode_steps
                    if self.pics:
                        for i, image in enumerate(episode_images):
                            writer_train.add_image('episode_{}'
                                                   .format(str(ep_print)), image.unsqueeze(0),
                                                   i)
    
                    if len(memory) > self.min_replay_size and ep_print > self.warm_up_episodes:
                        updates = self.learning_phase((last_episode_steps // 10) * 10 + 10, memory, updates,
                                                      writer_learn)
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

                # Security Wall, useful for longer training Phase
                while self.env.is_human_controlled():
                    pass

                # Let's test (if it is the case)
                if i_episode % self.eval_every == 0 and self.eval and i_episode != 0 and not restore:
                    # print('test')
                    self.test_phase(writer_test, i_run, updates)
                    # Wait for the human to leave the command
                    while self.env.is_human_controlled():
                        pass

                # TODO: HP Checkpoint and check correctness of checkpoint restoring
                if i_episode % self.eval_every == 0 and i_episode != 0 and not restore:
                    self.logger.important("Saving context...")
                    self.logger.info("To restart from here set this flag: --restore " + self.folder)
                    # Save Replay, net weights, hp, i_episode and i_run
                    with open(self.folder + "memory.pkl", "wb") as pickle_out:
                        memory.dump(pickle_out)
                    with open(self.folder + "context.json", "w+") as pickle_out:
                        json.dump((i_episode, i_run, updates, total_numsteps, running_episode_reward,
                                   running_episode_reward_100, last_episode_steps, episode_reward,
                                   episode_steps,
                                   timing, total_timing), pickle_out)
                    with open(self.folder + "rewards.pkl", "wb") as pickle_out:
                        pickle.dump(rewards, pickle_out)
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
                        episode_images.append(state_buffer.get_tensor()[0])
    
                    if i_episode < self.warm_up_episodes or len(memory) < self.min_replay_size:
                        # Warm_up phase -> Completely random choice of an action
                        action = self.env.action_space.sample()
                    else:
                        # Training phase -> Action sampled from policy
                        action = self.select_action(state, i_episode)

                    assert action.shape == self.env.action_space.shape
                    assert action is not None
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
                print("Memory {}/{}".format(len(memory), self.replay_size))
                timing = time.time() - ts
                total_timing = time.time() - in_ts
                start_episode = 0
                i_episode += 1
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
        
        actor_path = model_f + f"ddpg_actor_{env_name}_episode{i_episode}"
        critic_path = model_f + f"ddpg_critic_{env_name}_episode{i_episode}"
        torch.save(self.actor_net.state_dict(), actor_path)
        torch.save(self.critic_net.state_dict(), critic_path)

    # Load model parameters
    def load_model_to_play(self, env_name, folder, i_run, i_episode, suffix=""):
        model_f = folder + f'run_{i_run}/' + 'models/' + f"episode_{i_episode}/"
        if not os.path.exists(model_f):
            os.makedirs(model_f)
    
        actor_path = model_f + f"ddpg_actor_{env_name}_episode{i_episode}"
        critic_path = model_f + f"ddpg_critic_{env_name}_episode{i_episode}"
        self.load_model(actor_path, critic_path)
    
    # Load model parameters
    def load_model(self, actor_path, critic_path):
        if actor_path is not None:
            self.actor_net.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic_net.load_state_dict(torch.load(critic_path))
    
    # Backup model parameters
    def backup_model(self):
        model_f = self.folder + 'backup/'
        if not os.path.exists(model_f):
            os.makedirs(model_f)
        
        actor_path = model_f + f"ddpg_actor"
        actor_path_t = model_f + f"ddpg_actor_t"
        critic_path = model_f + f"ddpg_critic"
        critic_t_path = model_f + f"ddpg_critic_t"
        torch.save(self.actor_net.state_dict(), actor_path)
        torch.save(self.target_policy_net.state_dict(), actor_path_t)
        torch.save(self.critic_net.state_dict(), critic_path)
        torch.save(self.target_value_net.state_dict(), critic_t_path)
    
    # Restore model parameters
    def restore_model(self):
        model_f = self.folder + 'backup/'
        actor_path = model_f + f"ddpg_actor"
        actor_path_t = model_f + f"ddpg_actor_t"
        critic_path = model_f + f"ddpg_critic"
        critic_t_path = model_f + f"ddpg_critic_t"
        if actor_path is not None:
            self.actor_net.load_state_dict(torch.load(actor_path))
        if actor_path_t is not None:
            self.target_policy_net.load_state_dict(torch.load(actor_path_t))
        if critic_path is not None:
            self.critic_net.load_state_dict(torch.load(critic_path))
        if critic_t_path is not None:
            self.target_value_net.load_state_dict(torch.load(critic_t_path))
    
    def test_phase(self, writer_test, i_run, i_episode):
        n_tests = 0
        ts = time.time()
        rewards = []
        while n_tests < self.eval_episode:
        
            episode_reward = self.do_one_test()
            while self.env.is_human_controlled():
                pass
            if self.env.is_forget_enabled():
                self.logger.info("Last Test Episode Forgotten")
                self.env.reset_forget()
            else:
                rewards.append(episode_reward)
                n_tests += 1
    
        rewards = np.array(rewards)
        writer_test.add_scalar('test/average_reward', rewards.mean(), i_episode)
        writer_test.add_scalar('test/max_reward', rewards.max(), i_episode)
        writer_test.add_scalar('test/min_reward', rewards.min(), i_episode)
        writer_test.add_scalar('test/stdev_reward', rewards.std(), i_episode)
    
        self.logger.info("----------------------------------------")
        self.logger.info("Test {} ep.: {}, m_r: {}, max: {}, min: {}, std: {} time_spent {}s"
                         .format(self.eval_episode,
                                 i_episode,
                                 round(rewards.mean(), 2),
                                 round(rewards.max(), 2),
                                 round(rewards.min(), 2),
                                 round(rewards.std(), 2),
                                 round(time.time() - ts, 2)))
        self.save_model(self.env_name, self.folder + f"run_{i_run}/", i_episode)
        self.logger.info('Saving models...')
        self.logger.info("----------------------------------------")
    
    def learning_phase(self, updates_per_episode, memory, updates, writer_learn):
        time_update = time.time()
        # Let's update our parameters, this is the main part of learning
        for i in range(updates_per_episode):
            # Update parameters of all the networks
            actor_loss, critic_loss = self.update_parameters(memory, self.batch_size, updates)
            writer_learn.add_scalar('loss/actor', actor_loss, updates)
            writer_learn.add_scalar('loss/critic', critic_loss, updates)
            
            updates += 1
        # print(updates)
        self.logger.info("Update (up. {})took {}s"
                         .format(updates_per_episode,
                                 round(time.time() - time_update, 2)))
        return updates
    
    def print_nets(self, writer_train: SummaryWriter, ep_print: int):
        for k, v in self.actor_net.state_dict().items():
            # print(k)
            if (k.endswith('bias') or k.endswith('weight')) and (k.startswith('conv') or k.startswith('conv')):
                writer_train.add_histogram('policy/' + k, v, global_step=ep_print)
        for k, v in self.target_policy_net.state_dict().items():
            # print(k)
            if (k.endswith('bias') or k.endswith('weight')) and (k.startswith('conv') or k.startswith('conv')):
                writer_train.add_histogram('policy/' + k, v, global_step=ep_print)
        for k, v in self.critic_net.state_dict().items():
            if (k.endswith('bias') or k.endswith('weight')) and (k.startswith('conv') or k.startswith('conv')):
                writer_train.add_histogram('critic/' + k, v, global_step=ep_print)
        for k, v in self.target_value_net.state_dict().items():
            if (k.endswith('bias') or k.endswith('weight')) and (k.startswith('conv') or k.startswith('conv')):
                writer_train.add_histogram('critic_target/' + k, v, global_step=ep_print)
        
        pass
    
    def play(self):
        n_tests = 0
        ts = time.time()
        total_reward = 0
        while True:
            episode_reward = self.do_one_test()
            while self.env.is_human_controlled():
                pass
            if self.env.is_forget_enabled():
                self.logger.info("Last Test Episode Forgotten")
                self.env.reset_forget()
            else:
                n_tests += 1
                total_reward += episode_reward
                self.logger.info("Test {}, mean_r: {}, time_spent {}s"
                                 .format(n_tests,
                                         round(total_reward / n_tests, 2),
                                         round(time.time() - ts, 2)))
