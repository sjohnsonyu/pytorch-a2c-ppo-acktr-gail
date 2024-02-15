import copy
import glob
import os
import time
from collections import deque
from datetime import datetime

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pandas as pd

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate

from util.video_writer import VideoWriter


def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    timestr = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    exp_name = f'{args.env_name}-{timestr}'

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False, args)

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy,
                     'rnn_type': args.rnn_type,
                     'hidden_size': args.hidden_size
                     })
    actor_critic.to(device)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
        
    # # quick eval
    # exp_name = '2024-02-13_11-59-14'
    # actor_critic = torch.load(f'trained_models/ppo/FishEnv-v1-{exp_name}.pt',
    #                         map_location=torch.device('cpu')
    #                         )[0]
    # eval_with_video(args, device, actor_critic, exp_name + 'temp')
    training_loop(args, device, actor_critic, agent, envs, exp_name, eval_log_dir)
    eval_with_video(args, device, actor_critic, exp_name)


def training_loop(args, device, actor_critic, agent, envs, exp_name, eval_log_dir):
    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
            ], os.path.join(save_path, exp_name + ".pt"))  # TODO change to saving the best model

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            obs_rms = utils.get_vec_normalize(envs).obs_rms
            evaluate(actor_critic, obs_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)


# adapted from https://github.com/BruntonUWBio/plumetracknets/blob/main/code/ppo/main.py
def eval_lite(env, args, device, actor_critic):
    t_start = time.time()
    episode_summaries = []
    num_episodes = 0
    for i_episode in range(args.num_test_trials):
        recurrent_hidden_states = torch.zeros(1, 
                    actor_critic.recurrent_hidden_state_size, device=device)
        masks = torch.zeros(1, 1, device=device)
        obs = env.reset()

        reward_sum = 0
        ep_step = 0
        ep_step = 0

        while True:
            with torch.no_grad():
                # value, action, _, recurrent_hidden_states, activity = actor_critic.act(
                value, action, _, recurrent_hidden_states = actor_critic.act(
                    obs, 
                    recurrent_hidden_states, 
                    masks, 
                    deterministic=True)

            obs, reward, done, info = env.step(action)
            masks.fill_(0.0 if done else 1.0)

            reward_sum += reward.detach().numpy().squeeze()
            ep_step += 1

            if done:
                num_episodes += 1
                episode_summary = {
                    'idx': i_episode,
                    'reward_sum': reward_sum,
                    'n_steps': ep_step,
                }
                episode_summaries.append(episode_summary)
                break # out of while loop

    episode_summaries = pd.DataFrame(episode_summaries)
    r_mean = episode_summaries['reward_sum'].mean()
    r_std = episode_summaries['reward_sum'].std()
    comp_time = time.time() - t_start        
    steps_mean = episode_summaries['n_steps'].mean()
    eval_record = {
        'r_mean': np.around(r_mean, decimals=2),
        'r_std': np.around(r_std, decimals=2),
        'steps_mean': np.around(steps_mean, decimals=2),
        't': np.around(comp_time, decimals=2),
    }
    return eval_record


def eval_with_video(args, device, actor_critic, exp_name):
    env = make_vec_envs(args.env_name,
                        args.seed,
                        1,  # num_processes
                        args.gamma,
                        args.log_dir,
                        device,
                        True,
                        args
                        )
    # actor_critic = torch.load(f'trained_models/ppo/FishEnv-v1-{exp_name}.pt',
    #                           map_location=torch.device('cpu')
    #                           )[0]
    eval_record = eval_lite(env, args, device, actor_critic)
    print('eval_record', eval_record)

    video_writer = VideoWriter(f'{exp_name}', fps=10)
    print("Rendering video...")

    recurrent_hidden_states = torch.zeros(1, 
                actor_critic.recurrent_hidden_state_size, device=device)
    masks = torch.zeros(1, 1, device=device)
    obs = env.reset()

    while True:
        with torch.no_grad():
            frame = env.render(mode="rgb_array")
            video_writer.add_frame(frame)
            value, action, _, recurrent_hidden_states = actor_critic.act(
                obs,
                recurrent_hidden_states, 
                masks, 
                deterministic=True)
            print('obs', obs)
            print('action', action[-1])

        obs, _, done, _ = env.step(action)
        masks.fill_(0.0 if done else 1.0)
        if done:
            break

    # breakpoint()
    video_writer.save_video()



if __name__ == "__main__":
    main()
