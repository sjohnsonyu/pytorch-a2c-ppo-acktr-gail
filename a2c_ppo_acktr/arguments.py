import argparse

import torch
import sys
sys.path.append('./')
from util.constants import *


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--algo', default='a2c', help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument(
        '--gail',
        action='store_true',
        default=False,
        help='do imitation learning with gail')
    parser.add_argument(
        '--gail-experts-dir',
        default='./gail_experts',
        help='directory that contains expert demonstrations for gail')
    parser.add_argument(
        '--gail-batch-size',
        type=int,
        default=128,
        help='gail batch size (default: 128)')
    parser.add_argument(
        '--gail-epoch', type=int, default=5, help='gail epochs (default: 5)')
    parser.add_argument(
        '--lr', type=float, default=7e-4, help='learning rate (default: 7e-4)')
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.99,
        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--use-gae',
        action='store_true',
        default=False,
        help='use generalized advantage estimation')
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.95,
        help='gae lambda parameter (default: 0.95)')
    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.01,
        help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help='max norm of gradients (default: 0.5)')
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument(
        '--num-processes',
        type=int,
        default=16,
        help='how many training CPU processes to use (default: 16)')
    parser.add_argument(
        '--num-steps',
        type=int,
        default=5,
        help='number of forward steps in A2C (default: 5)')
    parser.add_argument(
        '--ppo-epoch',
        type=int,
        default=4,
        help='number of ppo epochs (default: 4)')
    parser.add_argument(
        '--num-mini-batch',
        type=int,
        default=32,
        help='number of batches for ppo (default: 32)')
    parser.add_argument(
        '--clip-param',
        type=float,
        default=0.2,
        help='ppo clip parameter (default: 0.2)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='log interval, one log per n updates (default: 10)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=100,
        help='save interval, one save per n updates (default: 100)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=None,
        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument(
        '--num-env-steps',
        type=int,
        default=10e6,
        help='number of environment steps to train (default: 10e6)')
    parser.add_argument(
        '--env-name',
        default='PongNoFrameskip-v4',
        help='environment to train on (default: PongNoFrameskip-v4)')
    parser.add_argument(
        '--log-dir',
        default='/tmp/gym/',
        help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument(
        '--save-dir',
        default='./trained_models/',
        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    parser.add_argument(
        '--use-proper-time-limits',
        action='store_true',
        default=False,
        help='compute returns taking into account time limits')
    parser.add_argument(
        '--recurrent_policy',
        action='store_true',
        default=False,
        help='use a recurrent policy')
    parser.add_argument(
        '--use-linear-lr-decay',
        action='store_true',
        default=False,
        help='use a linear schedule on the learning rate')
    

    ####### Custom arguments #######
    parser.add_argument("--simulation_mode", type=str, default="video", choices=["interactive", "video"], help='simulation mode: "interactive" or "video"')
    parser.add_argument("--agent", type=str, default="simple", choices=["simple", "ppo"], help='Agent to use: "simple" or "ppo"')
    parser.add_argument("--train", type=int, default=0, help='Train PPO model')
    parser.add_argument("--model_path", type=str, help='Path of PPO model if already trained')
    parser.add_argument("--num_train_steps", type=int, default=500_000, help='Number of steps to train PPO model')
    parser.add_argument("--num_test_steps", type=int, default=100, help='Number of steps to test PPO model')
    # parser.add_argument("--seed", type=int, default=0, help='Seed for environment')
    parser.add_argument("--num_test_trials", type=int, default=1, help='Number of test trials to run')
    parser.add_argument("--num_val_trials", type=int, default=1, help='Number of validation trials to run')
    parser.add_argument("--agent_init_pos_mode", type=str, default="random", choices=["random", "corner", "fixed_corner"], help="Initial position of agent")
    parser.add_argument("--sensing_radius", type=float, default=DEFAULT_SENSING_RADIUS, help="Agent sensing radius")
    parser.add_argument("--eating_radius", type=float, default=DEFAULT_EATING_RADIUS, help="Agent eating radius")
    parser.add_argument("--eod_cost", type=float, default=DEFAULT_EOD_COST, help="Agent EOD cost")
    parser.add_argument("--move_cost", type=float, default=DEFAULT_MOVE_COST, help="Agent move cost")
    parser.add_argument("--turn_cost", type=float, default=DEFAULT_TURN_COST, help="Agent turn cost")
    parser.add_argument("--num_food", type=int, default=10, help="Number of food in the arena")
    parser.add_argument("--observation_mode", type=str, default="vector", choices=["distance", "vector", "angle_dist", "angle_wall"], help="Observation mode for agent")
    parser.add_argument("--motion_mode", type=str, default="simple", choices=["simple", "kinetic"], help="Motion mode for agent")
    parser.add_argument("--reward_mode", type=str, default="food_distance_shaping", choices=["food_distance_shaping", "food_eaten", "hunger_meter"], help="Reward mode for agent")
    parser.add_argument("--food_init_pos_mode", type=str, default="random_with_close", choices=["random", "random_with_close"], help="Food position mode")
    parser.add_argument("--food_motion_mode", type=str, default="stationary", choices=["stationary", "random"], help="Food position mode")
    parser.add_argument("--eating_mode", type=str, default="auto", choices=["auto", "manual"], help="Eating mode for agent")
    parser.add_argument("--food_reward", type=float, default=DEFAULT_FOOD_REWARD, help="Reward for eating food")
    parser.add_argument("--max_linear_velocity", type=float, default=DEFAULT_MAX_LINEAR_VELOCITY, help="Maximum linear velocity for agent")
    parser.add_argument("--max_angular_velocity", type=float, default=DEFAULT_MAX_ANGULAR_VELOCITY, help="Maximum angular velocity for agent")
    parser.add_argument("--max_linear_accel", type=float, default=DEFAULT_MAX_LINEAR_ACCEL, help="Maximum linear acceleration for agent")
    parser.add_argument("--max_angular_accel", type=float, default=DEFAULT_MAX_ANGULAR_ACCEL, help="Maximum angular acceleration for agent")
    parser.add_argument("--egocentric_obs", type=int, default=0, help="Use egocentric observation mode")
    parser.add_argument("--food_renewal", type=int, default=0, help="Spawn new food when food eaten")

    parser.add_argument("--rnn_type", type=str, default="rnn", choices=["rnn", "gru"], help="RNN type for PPO model")
    parser.add_argument("--hidden_size", type=int, default=64, help="Hidden size for PPO model")

    parser.add_argument("--train_max_episode_steps", type=int, default=200, help="Maximum episode length for training")
    parser.add_argument(
        '--load_from_expname',
        default=None,
        help='experiment name to load from')
    parser.add_argument(
        '--eval_graphing',
        action='store_true',
        default=False,
        help='graph the evaluation')
    parser.add_argument("--arena_width", type=int, default=60, help="Width of arena in cm")
    parser.add_argument("--arena_height", type=int, default=60, help="Height of arena in cm")
    parser.add_argument("--video_fps", type=int, default=10, help="FPS for video recording")
    parser.add_argument("--init_energy_level", type=float, default=0.5, help="Starting energy level")


    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    assert args.algo in ['a2c', 'ppo', 'acktr']
    if args.recurrent_policy:
        assert args.algo in ['a2c', 'ppo'], \
            'Recurrent policy is not implemented for ACKTR'

    return args
