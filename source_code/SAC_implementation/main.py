import argparse
import datetime
import itertools
import json
import os
import time

import gym
import math
import numpy as np
import torch
from tensorboardX import SummaryWriter

from image_wrapper import ImageWrapper
from my_logging import Log
from normalized_actions import NormalizedActions
from replay_memory import ReplayMemory
from sac import SAC
from state_buffer import StateBuffer


class LoadFromFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        with values as f:
            parser.parse_args(f.read().split(), namespace)


def initial_setup():
    """Default parameters"""
    # Environment
    env_name = "HalfCheetah-v2"
    seed = math.floor(time.time())
    
    # Evaluation
    eval = True
    eval_every = 5
    eval_episode = 10
    
    # Net and SAC parameters
    policy = "Gaussian"
    gamma = 0.99
    tau = 0.005
    lr = 0.0003
    alpha = 0.2
    autotune_entropy = True
    hidden_size = 256
    img_size = 64
    
    # Episode
    warm_up_episode = 5
    num_episode = 200
    max_num_step = 200
    max_num_run = 20
    batch_size = 256
    replay_size = 1000000
    state_buffer_size = 2
    updates_per_step = 100
    target_update = 1
    
    parser = argparse.ArgumentParser(description='SAC Implementation with CNN or NN')
    parser.add_argument('--env_name', default=env_name, help='Name of the OpenAI Gym environment to run')
    parser.add_argument('--policy', default=policy, help='Gaussian | Deterministic policy to use in the algorithm')
    parser.add_argument('--eval', type=bool, default=eval, help='Enable eval of the learned policy')
    parser.add_argument('--eval_every', type=int, default=eval_every, help='Evaluate every X episodes')
    parser.add_argument('--eval_episode', type=int, default=eval_episode, help='Number of episode to test')
    parser.add_argument('--gamma', type=float, default=gamma, metavar='G', help='Discount factor for reward')
    parser.add_argument('--tau', type=float, default=tau, metavar='G', help='Tau coefficient (Target)')
    parser.add_argument('--lr', type=float, default=lr, metavar='G', help='learning rate')
    parser.add_argument('--alpha', type=float, default=alpha, metavar='G', help='Alpha Temperature parameter')
    parser.add_argument('--autotune_entropy', type=bool, default=autotune_entropy, metavar='G', help='Alpha Autotune')
    parser.add_argument('--seed', type=int, default=seed, metavar='N', help='Specify a Seed')
    parser.add_argument('--batch_size', type=int, default=batch_size, metavar='N', help='Batch size')
    parser.add_argument('--max_num_run', type=int, default=max_num_run, metavar='N', help='Max number of runs')
    parser.add_argument('--max_num_step', type=int, default=max_num_step, metavar='N', help='Max number of steps')
    parser.add_argument('--num_episode', type=int, default=num_episode, metavar='N', help='Max #episode per run')
    parser.add_argument('--hidden_size', type=int, default=hidden_size, metavar='N', help='Hidden size NN')
    parser.add_argument('--updates_per_step', type=int, default=updates_per_step, metavar='N',
                        help='#updates for each step')
    parser.add_argument('--warm_up_episode', type=int, default=warm_up_episode, metavar='N', help='Warm-Up episodes')
    parser.add_argument('--target_update', type=int, default=target_update, metavar='N', help='Target updates / update')
    parser.add_argument('--replay_size', type=int, default=replay_size, metavar='N', help='Size of replay buffer')
    parser.add_argument('--state_buffer_size', type=int, default=state_buffer_size, metavar='N',
                        help='Size of state buffer')
    parser.add_argument('--cuda', action="store_true", help='run on CUDA')
    parser.add_argument('--pics', action="store_true", help='run on Image')
    parser.add_argument('--img_size', type=int, default=img_size, metavar='N', help='Size of image (HW)')
    
    parser.add_argument('--load_from_json', type=str, default=None, help='Load From File')
    args = parser.parse_args()
    
    folder_ = '{}_SAC/'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.mkdir("./runs/" + folder_)
    logger_ = Log("./runs/" + folder_)
    if args.load_from_json is not None:
        try:
            argparse_dict = vars(args)
            with open(args.load_from_json) as data_file:
                data = json.load(data_file)
            argparse_dict.update(data)
        except FileNotFoundError:
            logger_.error("File not Valid")
            exit(1)
    
    return args, folder_, logger_


if __name__ == '__main__':
    in_ts = time.time()
    # Setting up Hyper-Parameters
    args, folder, logger = initial_setup()
    hp = vars(args)
    print("=== HYPERPARAMETERS ===")
    for key in hp:
        print(f"{key} : {hp[key]}")
    print("=======================")
    logger.debug("Initial setup completed.")
    # Create JSON of Hyper-Parameters for reproducibility
    with open("./runs/" + folder + "hp.json", 'w') as outfile:
        json.dump(vars(args), outfile)
    cnn = args.pics
    for i_run in range(args.max_num_run):
        logger.important(f"START TRAINING RUN {i_run}")
        
        # Make the environment
        env = gym.make(args.env_name)
        env._max_episode_steps = args.max_num_step
        env = NormalizedActions(env)
        if cnn:
            env = ImageWrapper(args.img_size, env)
        
        # Set Seed for repeatability
        torch.manual_seed(args.seed + i_run)
        np.random.seed(args.seed + i_run)
        env.seed(args.seed + i_run)
        env.action_space.np_random.seed(args.seed + i_run)
        
        # Setup the agent
        agent = SAC(args.state_buffer_size, env.action_space, args)
        
        # Setup TensorboardX
        writer_train = SummaryWriter(log_dir='runs/' + folder + 'run_' + str(i_run) + '/train')
        writer_test = SummaryWriter(log_dir='runs/' + folder + 'run_' + str(i_run) + '/test')
        
        # Setup Replay Memory
        memory = ReplayMemory(args.replay_size)
        
        # TRAINING LOOP
        total_numsteps = updates = running_episode_reward = running_episode_reward_100 = 0
        rewards = []
        
        for i_episode in itertools.count(1):
            print(updates)
            ts = time.time()
            episode_reward = episode_steps = 0
            done = False
            state = env.reset()
            if cnn:
                state_buffer = StateBuffer(args.state_buffer_size, state)
                state = state_buffer.get_state()
            
            critic_1_loss_acc = critic_2_loss_acc = policy_loss_acc = ent_loss_acc = alpha_acc = 0
            
            while not done:
                # if cnn:
                #     writer_train.add_images('episode_{}'.format(str(i_episode)), state_buffer.get_tensor(), episode_steps)
                if i_episode < args.warm_up_episode:
                    action = env.action_space.sample()  # Sample random action
                else:
                    action = agent.select_action(state)  # Sample action from policy
                
                next_state, reward, done, _ = env.step(action)  # Step
                env.render()
                if cnn:
                    state_buffer.push(next_state)
                    next_state = state_buffer.get_state()
                episode_steps += 1
                total_numsteps += 1
                episode_reward += reward
                # # Ignore the "done" signal if it comes from hitting the time horizon.
                # # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
                mask = 1 if done else float(not done)
                memory.push(state, action, reward, next_state, mask)  # Append transition to memory
                
                state = next_state
            if len(memory) > args.batch_size and i_episode > args.warm_up_episode:
                # Number of updates per step in environment
                # Update parameters of all the networks
                updates = agent.learning_phase(args.updates_per_step, memory, updates, writer_train, args.batch_size)

            rewards.append(episode_reward)
            running_episode_reward += (episode_reward - running_episode_reward) / i_episode
            if len(rewards) < 100:
                running_episode_reward_100 = running_episode_reward
            else:
                last_100 = rewards[-100:]
                running_episode_reward_100 = np.array(last_100).mean()
            writer_train.add_scalar('reward/train', episode_reward, i_episode)
            writer_train.add_scalar('reward/running_mean', running_episode_reward, i_episode)
            writer_train.add_scalar('reward/running_mean_last_100', running_episode_reward_100, i_episode)
            logger.info(
                "Ep. {}/{}, t {}, r_t {}, 100_mean {}, time_spent {}s | {}s ".format(i_episode, args.num_episode,
                                                                                     episode_steps,
                                                                                     round(episode_reward, 2),
                                                                                     round(running_episode_reward_100,
                                                                                           2),
                                                                                     round(time.time() - ts, 2),
                                                                                     str(datetime.timedelta(
                                                                                         seconds=time.time() - in_ts))))
            
            if updates % args.eval_every == 0 and args.eval and updates != 0:
                ts = time.time()
                total_reward = 0
                for _ in range(args.eval_episode):
                    old = env.reset()
                    state_buffer = StateBuffer(args.state_buffer_size, old)
                    episode_reward = 0
                    done = False
                    while not done:
                        state = state_buffer.get_state()
                        action = agent.select_action(state, eval=True)
                        
                        next_state, reward, done, _ = env.step(action)
                        env.render()
                        episode_reward += reward
                        
                        state_buffer.push(next_state)
                    total_reward += episode_reward
                
                writer_test.add_scalar('reward/test', total_reward / args.eval_episode, updates)
                
                logger.info("----------------------------------------")
                logger.info(
                    f"Test {args.eval_episode} step: {updates}, mean_r: {round(total_reward / args.eval_episode, 2)}"
                    f", time_spent {round(time.time() - ts, 2)}s")
                agent.save_model(args.env_name, "./runs/" + folder + f"run_{i_run}/", updates)
                logger.info('Saving models...')
                logger.info("----------------------------------------")
                
            if i_episode >= args.num_episode:
                break
        
        env.close()
