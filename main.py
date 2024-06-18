import copy
import glob
import os
import time
from collections import deque
from datetime import datetime
import json
import pickle

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
from evaluation import evaluate, eval_lite, eval_with_video, behavioral_assay

from util.video_writer import VideoWriter
from util.graphing_utils import *

def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    timestr = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if args.load_from_expname is None:
        exp_name = f'{args.env_name}-{timestr}'
    else:
        print('loading from expname:', args.load_from_expname, flush=True)
        exp_name = f'{args.env_name}-{args.load_from_expname}'

    log_dir = os.path.expanduser(args.log_dir)
    # eval_log_dir = log_dir + "_eval"
    # utils.cleanup_log_dir(log_dir)
    # utils.cleanup_log_dir(eval_log_dir)
    training_log = None
    eval_log = None

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs = make_vec_envs(args.env_name,
                         args.seed,
                         args.num_processes,
                         args.gamma,
                         args.log_dir,
                         device,
                         False,
                         args)
    
    eval_env = make_vec_envs(
        args.env_name,
        args.seed + 1000,
        num_processes=1,
        gamma=args.gamma, 
        log_dir=args.log_dir, 
        device=device,
        args=args,
        allow_early_resets=True)


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
        
    # Save args and config info
    # https://stackoverflow.com/questions/16878315/what-is-the-right-way-to-treat-argparse-namespace-as-a-dictionary

    if args.load_from_expname is None:
        args_filename = f"{args.save_dir}/{exp_name}_args.json"
        with open(args_filename, 'w') as fp:
            json.dump(vars(args), fp, indent=4)
        training_log, eval_log = training_loop(args, device, actor_critic, agent, envs, exp_name, training_log=training_log, eval_log=eval_log, eval_env=eval_env)
        original_exp_name = None
    else:
        actor_critic, _ = torch.load(f'trained_models/ppo/{exp_name}.pt')
        actor_critic.to(device)
        original_exp_name = exp_name
        exp_name = exp_name + f'_rerun_{timestr}'

    behavioral_assay(args, device, actor_critic, exp_name, original_exp_name, 3, num_pos_divisions=2)
    eval_with_video(args, device, actor_critic, exp_name, original_exp_name=original_exp_name)


def training_loop(args,
                  device,
                  actor_critic,
                  agent,
                  envs,
                  exp_name,
                  training_log=None,
                  eval_log=None,
                  eval_env=None,
                  ):
    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)
    best_mean = 0.0

    training_log = training_log if training_log is not None else []
    eval_log = eval_log if eval_log is not None else []

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
                value, action, action_log_prob, recurrent_hidden_states, activities = actor_critic.act(
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
            ], os.path.join(save_path, exp_name + ".pt"))

            current_mean = np.median(episode_rewards)
            # Save best!
            if current_mean >= best_mean:
                best_mean = current_mean
                fname = os.path.join(save_path, f'{exp_name}_best.pt')
                torch.save([
                    actor_critic,
                    getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
                ], fname)
                print('Saved', fname)

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print("Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}"
                .format(j,
                        total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards),
                        np.mean(episode_rewards),
                        np.median(episode_rewards),
                        np.min(episode_rewards),
                        np.max(episode_rewards)),
                flush=True)

            training_log.append({
                    'update': j,
                    'total_updates': num_updates,
                    'T': total_num_steps,
                    'FPS': int(total_num_steps / (end - start)),
                    'window': len(episode_rewards),
                    'mean': np.mean(episode_rewards),
                    'median': np.median(episode_rewards),
                    'min': np.min(episode_rewards),
                    'max': np.max(episode_rewards),
                    'std': np.std(episode_rewards),
                })

            # Save training curve
            save_path = os.path.join(args.log_dir, exp_name) #args.save_dir # os.path.join(args.save_dir, args.algo)
            os.makedirs(save_path, exist_ok=True)
            fname = os.path.join(save_path, f'{exp_name}_training_log.csv')
            pd.DataFrame(training_log).to_csv(fname)


        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            # obs_rms = utils.get_vec_normalize(envs).obs_rms
            # evaluate(actor_critic, obs_rms, args.env_name, args.seed,
            #          args.num_processes, eval_log_dir, device)

            if eval_env is not None:
                eval_record = eval_lite(eval_env, args, device, actor_critic, exp_name, args.num_val_trials, make_graphs=False)
                eval_record['T'] = total_num_steps
                eval_record['update'] = j
                eval_log.append(eval_record)
                print("eval_lite:", eval_record, flush=True)

                save_path = os.path.join(args.log_dir, exp_name) #args.save_dir # os.path.join(args.save_dir, args.algo)
                os.makedirs(save_path, exist_ok=True)
                fname = os.path.join(save_path, f'{exp_name}_eval_log.csv')
                pd.DataFrame(eval_log).to_csv(fname)

    return training_log, eval_log



if __name__ == "__main__":
    main()
