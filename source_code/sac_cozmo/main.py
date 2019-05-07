import argparse
import datetime
import itertools
import json
import os
import sys
import time

import cozmo
import gym
import gym_cozmo
import math
import numpy as np
import torch
from image_wrapper import ImageWrapper
from my_logging import Log
from normalized_actions import NormalizedActions
from replay_memory import ReplayMemory
from sac import SAC
from state_buffer import StateBuffer
from tensorboardX import SummaryWriter


class LoadFromFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        with values as f:
            parser.parse_args(f.read().split(), namespace)


def initial_setup():
    """Default parameters"""
    # Environment
    env_name = "CozmoDriver-v0"
    seed = math.floor(time.time())
    
    # Evaluation
    eval = True
    eval_every = 10
    eval_episode = 2
    
    # Net and SAC parameters
    policy = "Gaussian"
    gamma = 0.99
    tau = 0.005
    lr = 0.0003
    alpha = 0.2
    autotune_entropy = True
    hidden_size = 256
    img_size = 128
    
    # Episode
    warm_up_episode = 256
    num_episode = 200
    max_num_step = 200
    max_num_run = 20
    batch_size = 64
    replay_size = 1000000
    state_buffer_size = 1
    updates_per_step = 1
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
    parser.add_argument('--warm_up_steps', type=int, default=warm_up_episode, metavar='N', help='Warm-Up steps')
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


def run(sdk_conn):
    robot = sdk_conn.wait_for_robot()
    robot.enable_device_imu(True, True, True)
    
    # Turn on image receiving by the camera
    robot.camera.image_stream_enabled = True
    in_ts = time.time()
    # Setting up Hyper-Parameters
    args, folder, logger = initial_setup()
    logger.debug("Initial setup completed.")
    # Create JSON of Hyper-Parameters for reproducibility
    with open("./runs/" + folder + "hp.json", 'w') as outfile:
        json.dump(vars(args), outfile)
    cnn = args.pics
    gym_cozmo.register(
        id='CozmoDriver-v0',
        entry_point='gym_cozmo.envs:CozmoEnv',
        kwargs={'robot': robot})
    for i_run in range(args.max_num_run):
        logger.important(f"START TRAINING RUN {i_run}")
        
        # Make the environment
        
        env = gym.make('CozmoDriver-v0')
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
        i_episode = 0
        while True:
            robot.stop_all_motors()
            if i_episode > args.num_episode:
                break
            while env.is_human_controlled():
                continue
            ts = time.time()
            episode_reward = episode_steps = 0
            done = False
            info = {'undo': False}
            episode_replay = ReplayMemory(args.replay_size)
            state = env.reset()
            state_buffer = None
            if cnn:
                state_buffer = StateBuffer(args.state_buffer_size, state)
                state = state_buffer.get_state()
    
            critic_1_loss_acc = critic_2_loss_acc = policy_loss_acc = ent_loss_acc = alpha_acc = 0
    
            while not done:
                if cnn:
                    writer_train.add_image('episode_{}'.format(str(i_episode)), state_buffer.get_state(),
                                           episode_steps)
                if len(memory) < args.warm_up_steps:
                    action = env.action_space.sample()
                else:
                    action = agent.select_action(env, state)  # Sample action from policy
                    if len(memory) > args.batch_size:
                        # Number of updates per step in environment
                        for i in range(args.updates_per_step):
                            # Update parameters of all the networks
                            critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(
                                memory,
                                args.batch_size,
                                updates)
                    
                            critic_1_loss_acc += critic_1_loss
                            critic_2_loss_acc += critic_2_loss
                            policy_loss_acc += policy_loss
                            ent_loss_acc += ent_loss
                            alpha_acc += alpha
                            updates += 1
        
                next_state, reward, done, info = env.step(action)  # Step
                if cnn:
                    state_buffer.push(next_state)
                    next_state = state_buffer.get_state()
                episode_steps += 1
                total_numsteps += 1
                episode_reward += reward
                # # Ignore the "done" signal if it comes from hitting the time horizon.
                # # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
                mask = 1 if done else float(not done)
                episode_replay.push(state, action, reward, next_state, mask)  # Append transition to memory
        
                state = next_state
    
            print(len(memory))
            print(len(episode_replay))
            if not info['undo']:
                i_episode += 1
                memory.push_memory(episode_replay)
                print(len(memory))
        
                rewards.append(episode_reward)
                running_episode_reward += (episode_reward - running_episode_reward) / i_episode
                if len(rewards) < 100:
                    running_episode_reward_100 = running_episode_reward
                else:
                    last_100 = rewards[-100:]
                    running_episode_reward_100 = np.array(last_100).mean()
                writer_train.add_scalar('loss/critic_1', critic_1_loss_acc / episode_steps, i_episode)
                writer_train.add_scalar('loss/critic_2', critic_2_loss_acc / episode_steps, i_episode)
                writer_train.add_scalar('loss/policy', policy_loss_acc / episode_steps, i_episode)
                writer_train.add_scalar('loss/entropy_loss', ent_loss_acc / episode_steps, i_episode)
                writer_train.add_scalar('entropy_temperature/alpha', alpha_acc / episode_steps, i_episode)
                writer_train.add_scalar('reward/train', episode_reward, i_episode)
                writer_train.add_scalar('reward/running_mean', running_episode_reward, i_episode)
                writer_train.add_scalar('reward/running_mean_last_100', running_episode_reward_100, i_episode)
                logger.info(
                    "Ep. {}/{}, t {}, r_t {}, 100_mean {}, time_spent {}s | {}s ".format(i_episode, args.num_episode,
                                                                                         episode_steps,
                                                                                         round(episode_reward, 2),
                                                                                         round(
                                                                                             running_episode_reward_100,
                                                                                             2),
                                                                                         round(time.time() - ts, 2),
                                                                                         str(datetime.timedelta(
                                                                                             seconds=time.time() - in_ts))))
    
    
            if i_episode % args.eval_every == 0 and args.eval == True:
                ts = time.time()
                total_reward = 0
                for _ in range(args.eval_episode):
                    old = env.reset()
                    state_buffer = StateBuffer(args.state_buffer_size, old)
                    episode_reward = 0
                    done = False
                    while not done:
                        state = state_buffer.get_state()
                        action = agent.select_action(env, state, eval=True)
                
                        next_state, reward, done, _ = env.step(action)
                        env.render()
                        episode_reward += reward
                
                        state_buffer.push(next_state)
                    total_reward += episode_reward
        
                writer_test.add_scalar('reward/test', total_reward / args.eval_episode, i_episode)
        
                logger.info("----------------------------------------")
                logger.info(
                    f"Test {args.eval_episode} ep.: {i_episode}, mean_r: {round(total_reward / args.eval_episode, 2)}"
                    f", time_spent {round(time.time() - ts, 2)}s")
                agent.save_model(args.env_name, "./runs/" + folder + f"run_{i_run}/", i_episode)
                logger.info('Saving models...')
                logger.info("----------------------------------------")
        
        env.close()


if __name__ == '__main__':
    cozmo.setup_basic_logging()
    try:
        cozmo.connect(run)
    except KeyboardInterrupt as e:
        pass
    except cozmo.ConnectionError as e:
        sys.exit("A connection error occurred: %s" % e)

if __name__ == '__main__':
    in_ts = time.time()
    # Setting up Hyper-Parameters
    args, folder, logger = initial_setup()
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
                    if len(memory) > args.batch_size:
                        # Number of updates per step in environment
                        for i in range(args.updates_per_step):
                            # Update parameters of all the networks
                            critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(
                                memory,
                                args.batch_size,
                                updates)
                            
                            critic_1_loss_acc += critic_1_loss
                            critic_2_loss_acc += critic_2_loss
                            policy_loss_acc += policy_loss
                            ent_loss_acc += ent_loss
                            alpha_acc += alpha
                            updates += 1
                
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
            
            if i_episode > args.num_episode:
                break
            rewards.append(episode_reward)
            running_episode_reward += (episode_reward - running_episode_reward) / i_episode
            if len(rewards) < 100:
                running_episode_reward_100 = running_episode_reward
            else:
                last_100 = rewards[-100:]
                running_episode_reward_100 = np.array(last_100).mean()
            writer_train.add_scalar('loss/critic_1', critic_1_loss_acc / episode_steps, i_episode)
            writer_train.add_scalar('loss/critic_2', critic_2_loss_acc / episode_steps, i_episode)
            writer_train.add_scalar('loss/policy', policy_loss_acc / episode_steps, i_episode)
            writer_train.add_scalar('loss/entropy_loss', ent_loss_acc / episode_steps, i_episode)
            writer_train.add_scalar('entropy_temperature/alpha', alpha_acc / episode_steps, i_episode)
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
            
            if i_episode % args.eval_every == 0 and args.eval == True:
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
                
                writer_test.add_scalar('reward/test', total_reward / args.eval_episode, i_episode)
                
                logger.info("----------------------------------------")
                logger.info(
                    f"Test {args.eval_episode} ep.: {i_episode}, mean_r: {round(total_reward / args.eval_episode, 2)}"
                    f", time_spent {round(time.time() - ts, 2)}s")
                agent.save_model(args.env_name, "./runs/" + folder + f"run_{i_run}/", i_episode)
                logger.info('Saving models...')
                logger.info("----------------------------------------")
        
        env.close()
