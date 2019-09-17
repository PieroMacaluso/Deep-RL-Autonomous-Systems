import argparse
import datetime
import json
import logging
import os
import sys
import time

import cozmo
import gym
import gym_cozmo
import math
from tensorboard import program

from OUNoise import OUNoise
from my_logging import Log
from ddpg import DDPG


def initial_setup() -> (argparse.Namespace, str, Log, bool):
    """
    Initialization of default parameters and parsing of command line arguments.
    
    :return: arguments, name of the main folder of the experiment and logger.
    :rtype: (argparse.Namespace, str, Log, bool)
    """
    # Environment
    env_name = "CozmoDriver-v0"
    seed = math.floor(time.time())
    
    # Evaluation
    eval = True
    eval_every = 20
    eval_episode = 5
    
    # Net and DDPG parameters
    eps_start = 0.9
    eps_end = 0.2
    eps_decay = 1000
    
    # Noise
    mu = 0.0
    sigma = 0.3
    theta = 0.15
    
    # Net and SAC parameters
    gamma = 0.99
    tau = 0.005
    lr = 0.0003
    hidden_size = 256
    img_h = 64
    img_w = 64
    
    # Episode
    warm_up_episodes = 0
    num_episode = 500
    max_num_run = 5
    batch_size = 64
    replay_size = 5000
    min_replay_size = 64
    state_buffer_size = 2
    updates_per_episode = 100
    target_update = 1
    
    parser = argparse.ArgumentParser(description='SAC Implementation with CNN or NN')
    parser.add_argument('--env_name', default=env_name, help='Name of the OpenAI Gym environment to run')
    parser.add_argument('--eps_start', type=float, default=eps_start, help='eps_start')
    parser.add_argument('--eps_end', type=float, default=eps_end, help='eps_end')
    parser.add_argument('--eps_decay', type=float, default=eps_decay, help='eps_decay')
    parser.add_argument('-noise', nargs=3, default=[mu, sigma, theta], metavar=('mu', 'sigma', 'theta'), type=float,
                        help='Ornstein Uhlenbeck process noise parameters')
    parser.add_argument('--eval', type=bool, default=eval, help='Enable eval of the learned policy')
    parser.add_argument('--eval_every', type=int, default=eval_every, help='Evaluate every X episodes')
    parser.add_argument('--eval_episode', type=int, default=eval_episode, help='Number of episode to test')
    parser.add_argument('--gamma', type=float, default=gamma, metavar='G', help='Discount factor for reward')
    parser.add_argument('--tau', type=float, default=tau, metavar='G', help='Tau coefficient (Target)')
    parser.add_argument('--lr', type=float, default=lr, metavar='G', help='learning rate')
    parser.add_argument('--seed', type=int, default=seed, metavar='N', help='Specify a Seed')
    parser.add_argument('--batch_size', type=int, default=batch_size, metavar='N', help='Batch size')
    parser.add_argument('--max_num_run', type=int, default=max_num_run, metavar='N', help='Max number of runs')
    parser.add_argument('--num_episode', type=int, default=num_episode, metavar='N', help='Max #episode per run')
    parser.add_argument('--hidden_size', type=int, default=hidden_size, metavar='N', help='Hidden size NN')
    parser.add_argument('--updates_per_episode', type=int, default=updates_per_episode, metavar='N',
                        help='#updates for each step')
    parser.add_argument('--warm_up_episodes', type=int, default=warm_up_episodes, metavar='N', help='Warm-Up steps')
    parser.add_argument('--target_update', type=int, default=target_update, metavar='N', help='Target updates / update')
    parser.add_argument('--replay_size', type=int, default=replay_size, metavar='N', help='Size of replay buffer')
    parser.add_argument('--min_replay_size', type=int, default=min_replay_size, metavar='N',
                        help='Min Size of replay buffer')
    parser.add_argument('--state_buffer_size', type=int, default=state_buffer_size, metavar='N',
                        help='Size of state buffer')
    parser.add_argument('--cuda', action="store_true", help='run on CUDA')
    parser.add_argument('--pics', action="store_true", help='run on Image')
    parser.add_argument('--img_h', type=int, default=img_h, metavar='N', help='Size of image (H)')
    parser.add_argument('--img_w', type=int, default=img_w, metavar='N', help='Size of image (W)')
    parser.add_argument('--load_from_json', type=str, default=None, help='Load From File')
    parser.add_argument('--restore', type=str, default=None, help='Folder of experiment to restore')
    
    args = parser.parse_args()
    if args.restore:
        folder_ = args.restore
        restore = True
    else:
        folder_ = './runs/{}_SAC_CozmoDriver-v0/'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        restore = False
        os.mkdir(folder_)
    logger_ = Log(folder_)
    if args.load_from_json is not None:
        try:
            argparse_dict = vars(args)
            with open(args.load_from_json) as data_file:
                data = json.load(data_file)
            argparse_dict.update(data)
        except FileNotFoundError:
            logger_.error("File not Valid")
            exit(1)
    elif args.restore:
        try:
            argparse_dict = vars(args)
            with open(args.restore + "hp.json") as data_file:
                data = json.load(data_file)
            argparse_dict.update(data)
        except FileNotFoundError:
            logger_.error("File not Valid")
            exit(1)
    
    return args, folder_, logger_, restore


class TensorBoardTool:
    """
    Class used to initialize and start TensorBoardX.
    """
    
    def __init__(self, dir_path: str):
        """
        Constructor
        
        :param dir_path: path of TensorBoardX experiment files
        :type dir_path:  str
        """
        
        self.dir_path = dir_path
    
    def run(self) -> str:
        """
        Run TensorBoardX using the args specified in the code.
        
        :return: url
        :rtype: str
        """
        
        # Remove http messages
        log = logging.getLogger('werkzeug').setLevel(logging.ERROR)
        # Start tensorboard server
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', self.dir_path, '--host', 'localhost', '--samples_per_plugin', 'images=2000'
                                                                                                           ''])
        url = tb.launch()
        sys.stdout.write('TensorBoard at %s \n' % url)
        return url


def run(sdk_conn: cozmo.conn):
    """
    Container of the main loop. It is necessary to work with Cozmo. This is called by the cozmo.connect
    presents in the main loop of this file.
    
    :param sdk_conn: SDK connection to Anki Cozmo
    :type sdk_conn: cozmo.conn
    :return: nothing
    :rtype: nothing
    """
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace is not None and gettrace():
        debug = True
    else:
        debug = False
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    robot = sdk_conn.wait_for_robot()
    robot.enable_device_imu(True, True, True)
    # Turn on image receiving by the camera
    robot.camera.image_stream_enabled = True
    
    # Setting up Hyper-Parameters
    args, folder, logger, restore = initial_setup()
    if not debug:
        tb_tool = TensorBoardTool(folder)
        tb_tool.run()
    logger.debug("Initial setup completed.")
    
    # Create JSON of Hyper-Parameters for reproducibility
    with open(folder + "hp.json", 'w') as outfile:
        json.dump(vars(args), outfile)
    
    # Initialize Environment
    gym_cozmo.initialize(robot, args.img_h, args.img_w)
    env = gym.make(args.env_name)
    
    # Setup the agent
    agent = DDPG(args.state_buffer_size, env.action_space, env, args, folder, logger)
    agent.train(args.max_num_run, restore)
    env.close()
    logger.important("Program closed correctly!")


if __name__ == '__main__':
    # cozmo.setup_basic_logging()
    try:
        cozmo.connect(run)
    except KeyboardInterrupt as e:
        pass
    except cozmo.ConnectionError as e:
        sys.exit("A connection error occurred: %s" % e)
