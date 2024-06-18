import numpy as np
import torch

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs

from util.graphing_utils import *


# Sanity checking script to test that the environment is working for the given
# reward and observation modes, as well as the passed parameters. NOTE that
# this script does not test anything related to the model or training, only
# the environment.
def main():
    args = get_args() # can use default values to start with 
    device = torch.device("cuda:0" if args.cuda else "cpu")

    for reward_mode in ["food_eaten", "hunger_meter", "hunger_with_done", "hunger_metabolism"]:
        for observation_mode in ["angle_dist", "angle_wall", "angle_wall_hunger", "quad_dist_prob"]:
            print(reward_mode, observation_mode)
            args.reward_mode = reward_mode
            args.observation_mode = observation_mode

            try:
                env = make_vec_envs(
                    args.env_name,
                    args.seed + 1000,
                    num_processes=1,
                    gamma=args.gamma,
                    log_dir=args.log_dir,
                    device=device,
                    args=args,
                    allow_early_resets=True
                )
                env.reset()
                arbitrary_action = torch.Tensor(np.array([[1., 1., 1.]], dtype=np.float32))
                env.step(arbitrary_action)
                env.step(arbitrary_action)
            except Exception as e:
                print(e)
            print()
        


if __name__ == "__main__":
    main()
