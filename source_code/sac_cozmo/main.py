import argparse
import datetime
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
from tensorboardX import SummaryWriter

from image_wrapper import ImageWrapper
from my_logging import Log
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
    img_size = 84
    
    # Episode
    warm_up_steps = 1024
    num_episode = 500
    max_num_step = 200
    max_num_run = 20
    batch_size = 128
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
    parser.add_argument('--warm_up_steps', type=int, default=warm_up_steps, metavar='N', help='Warm-Up steps')
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
        kwargs={'robot': robot, 'image_dim': args.img_size})
    env = gym.make('CozmoDriver-v0')
    
    # Setup the agent
    agent = SAC(args.state_buffer_size, env.action_space, env, args, folder, logger)
    agent.train(args.max_num_run)


if __name__ == '__main__':
    cozmo.setup_basic_logging()
    try:
        cozmo.connect(run)
    except KeyboardInterrupt as e:
        pass
    except cozmo.ConnectionError as e:
        sys.exit("A connection error occurred: %s" % e)
