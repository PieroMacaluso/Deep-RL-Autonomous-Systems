import argparse
import datetime

import gc
import roboschool
import gym
import torch

from DDPG import DDPG
from ImageWrapper import ImageWrapper
from NormalizedActions import NormalizedActions
from OUNoise import OUNoise
from Simulation import Simulation


def get_args():
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
    # TODO: Revise this part
    parser = argparse.ArgumentParser(description='State implementation of DDPG.',
                                     epilog="No example of usage")
    parser.add_argument('-simulation_actor', nargs=2, default=[False, ''], metavar=('boolean', 'path'))
    parser.add_argument('-rec', nargs=1, default=False, metavar='boolean')
    parser.add_argument('-env', default='CarRacing-v0', type=str, help='Name of the Gym Environment')
    parser.add_argument('-noise', nargs=3, default=[0.0, 0.3, 0.15], metavar=('mu', 'sigma', 'theta'), type=float,
                        help='Ornstein Uhlenbeck process noise parameters')
    parser.add_argument('-eps', nargs=3, default=[0.9, 0.2, 300], metavar=('start', 'end', 'decay'), type=float,
                        help='Epsilon Decay process to decay the noise')
    parser.add_argument('-replay', nargs=3, default=[100, 2500, 10000000],
                        metavar=('batch_size', 'replay_min_size', 'replay_max_size'), type=int,
                        help='Replay Buffer parameters')
    parser.add_argument('-sim', nargs=2, default=[1000, 1000],
                        metavar=('n_episode', 'episode_max_len'), type=int, help='Loop Simulation parameters')
    parser.add_argument('-actor', nargs=3, default=[0, 'adam', 1e-4], metavar=('weight_decay', 'update_method', 'lr'))
    parser.add_argument('-critic', nargs=3, default=[0, 'adam', 1e-4], metavar=('weight_decay', 'update_method', 'lr'))
    parser.add_argument('-update', nargs=3, default=[0.99, 0.001, 1],
                        metavar=('discount', 'soft_target_tau', 'n_updates_per_sample'),
                        type=float, help='Update phase')
    parser.add_argument('-test', nargs=1, default=20,
                        metavar='eval_samples', type=int, help='Testing phase')
    parser.add_argument("-pics", type=str2bool, nargs='?',
                        const=True, default='yes',
                        help="Activate CNN Mode.")
    
    return parser.parse_args()


if __name__ == '__main__':
    str_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    assert torch.cuda.is_available(), "CUDA is not available"
    args = get_args()
    if args.simulation_actor[0]:
        env = NormalizedActions(gym.make(args.env))
        if args.rec:
            env = gym.wrappers.Monitor(env, "recording")
        sim = Simulation(env, args.simulation_actor[1])
        sim.simulate()
        env.close()
    else:
        for i in range(20):
            print("----------------------------------------")
            print("Run: {}".format(i))
            print("Algorithm: {}".format('DDPG'))
            print("Environment: {}".format(args.env))
            print("Model: {}".format('CNN' if args.pics else 'NN'))
            print("----------------------------------------")

            env = NormalizedActions(gym.make(args.env))
            test_env = NormalizedActions(gym.make(args.env))
            if args.pics:
                env = ImageWrapper(64, env)
                test_env = ImageWrapper(64, test_env)
            env.env._max_episode_steps = args.sim[1]
            test_env.env._max_episode_steps = args.sim[1]
            ou_noise = OUNoise(env.action_space, mu=args.noise[0], sigma=args.noise[1], theta=args.noise[2])
            ddpg = DDPG(env, test_env, exp_strategy=ou_noise,
                        eps_start=args.eps[0], eps_end=args.eps[1], eps_decay=args.eps[2],
                        batch_size=args.replay[0], replay_min_size=args.replay[1], replay_max_size=args.replay[2],
                        n_episode=args.sim[0], episode_max_len=args.sim[1],
                        actor_weight_decay=args.actor[0], actor_update_method=args.actor[1], actor_lr=args.actor[2],
                        critic_weight_decay=args.critic[0], critic_update_method=args.critic[1], critic_lr=args.critic[2],
                        size_state_buffer=3, timestamp=str_time,
                        discount=args.update[0], soft_target_tau=args.update[1], n_updates_per_sample=args.update[2],
                        eval_samples=args.test, run=i, pics=args.pics)
            ddpg.train()
            env.close()
            gc.collect()
