import os
import json
import time
import pickle

import numpy as np
import torch
import pandas as pd

from util.graphing_utils import *
from util.video_writer import VideoWriter


from a2c_ppo_acktr import utils
from a2c_ppo_acktr.envs import make_vec_envs


def evaluate(actor_critic, obs_rms, env_name, seed, num_processes, eval_log_dir,
             device):
    eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes,
                              None, eval_log_dir, device, True)

    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.obs_rms = obs_rms

    eval_episode_rewards = []

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)

    while len(eval_episode_rewards) < 10:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=True)

        # Obser reward and next obs
        obs, _, done, infos = eval_envs.step(action)

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])

    eval_envs.close()

    print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards)))


# adapted from https://github.com/BruntonUWBio/plumetracknets/blob/main/code/ppo/main.py
def eval_lite(env, args, device, actor_critic, exp_name, num_eval_episodes, make_graphs=False, original_exp_name=None):
    t_start = time.time()
    episode_summaries = []
    all_obs_history = []
    all_eod_history = []
    all_action_history = []
    episode_logs = []
    all_activities = []
    num_episodes = 0
    for i_episode in range(num_eval_episodes):
        recurrent_hidden_states = torch.zeros(1,
                    actor_critic.recurrent_hidden_state_size, device=device)
        masks = torch.zeros(1, 1, device=device)
        obs = env.reset()

        reward_sum = 0
        ep_step = 0

        episode_activities = []

        while True:
            with torch.no_grad():
                # breakpoint()
                value, action, _, recurrent_hidden_states, activities = actor_critic.act(
                    obs, 
                    recurrent_hidden_states,
                    masks,
                    deterministic=True)
                episode_activities.append(activities)

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
                episode_logs.append(info[0])
                all_obs_history.append(info[0]['observations'])
                all_eod_history.append(info[0]['eods'])
                all_action_history.append(info[0]['actions'])
                all_activities.append(episode_activities)
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
    if make_graphs:
        def make_figures_directory(exp_name):
            directory = f"figures/{exp_name}"
            if not os.path.exists(directory):
                os.makedirs(directory)

        # LOGS
        is_rerun = original_exp_name is not None
        original_exp_name = exp_name if original_exp_name is None else original_exp_name
        save_path = os.path.join(args.log_dir, original_exp_name) #args.save_dir # os.path.join(args.save_dir, args.algo)
        os.makedirs(save_path, exist_ok=True)
        test_log_filename = os.path.join(save_path, f'{exp_name}_test_log.csv')
        # pd.DataFrame(eval_record).to_csv(test_log_filename)
        with open(test_log_filename, 'w') as f:
            json.dump(eval_record, f, indent=4)

        with open(os.path.join(save_path, f'{exp_name}_episode_logs.pkl'), 'wb') as f:
            pickle.dump(episode_logs, f)

        def to_cpu(obj):
            if torch.is_tensor(obj):
                return obj.cpu()
            elif isinstance(obj, dict):
                return {k: to_cpu(v) for k, v in obj.items()}
            elif isinstance(obj, dict):
                return {k: to_cpu(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_cpu(v) for v in obj]
            else:
                return obj

        activities_cpu = to_cpu(all_activities)
        with open(os.path.join(save_path, f'{exp_name}_activities.pkl'), 'wb') as f:
            pickle.dump(activities_cpu, f)

        figures_directory = f"figures/{original_exp_name}"
        if is_rerun:
            figures_directory = os.path.join(figures_directory, exp_name)
        if not os.path.exists(figures_directory):
            os.makedirs(figures_directory)

        plot_distance_vs_eod_rate(np.concatenate(all_eod_history), np.concatenate(all_obs_history), figures_directory, exp_name)
        plot_distance_vs_eod_spi(all_eod_history, all_obs_history, figures_directory, exp_name)  # FIXME slightly hacky
        plot_spi_distribution(all_eod_history, figures_directory, exp_name)

        try:
            training_log = pd.read_csv(f'{args.log_dir}/{original_exp_name}/{original_exp_name}_training_log.csv', index_col=0)
            eval_log = pd.read_csv(f'{args.log_dir}/{original_exp_name}/{original_exp_name}_eval_log.csv', index_col=0)
            plot_training_val_rewards(training_log, eval_log, original_exp_name)
        except Exception as e:
            print("Problem with plotting training and val rewards: ", e, flush=True)

    return eval_record


def eval_with_video(args, device, actor_critic, exp_name, original_exp_name):
    args.min_train_init_energy_level = None
    args.max_train_init_energy_level = None
    args.min_train_num_food = None
    args.max_train_num_food = None
    env = make_vec_envs(args.env_name,
                        args.seed,
                        1,  # num_processes
                        args.gamma,
                        args.log_dir,
                        device,
                        True,
                        args
                        )
    eval_record = eval_lite(env, args, device, actor_critic, exp_name, args.num_test_trials, make_graphs=args.eval_graphing, original_exp_name=original_exp_name)
    print('eval_record', eval_record, flush=True)

    video_writer = VideoWriter(f'{exp_name}', fps=args.video_fps)
    print("Rendering video...", flush=True)

    recurrent_hidden_states = torch.zeros(1,
                actor_critic.recurrent_hidden_state_size, device=device)
    masks = torch.zeros(1, 1, device=device)
    obs = env.reset()

    episode_activities = []

    while True:
        with torch.no_grad():
            frame = env.render(mode="rgb_array")
            video_writer.add_frame(frame)
            value, action, _, recurrent_hidden_states, activities = actor_critic.act(
                obs,
                recurrent_hidden_states,
                masks,
                deterministic=True)

            episode_activities.append(activities['rnn_hxs'].numpy().squeeze())

        obs, _, done, _ = env.step(action)
        masks.fill_(0.0 if done else 1.0)
        if done:
            break

    activity_frames, _ = render_activities(episode_activities, episode_idx=None, pca_dims=3, shared_pca=None)
    for frame in activity_frames:
        video_writer.add_frame(frame, track=1)

    video_writer.save_video()
    video_writer.save_stitched_video()


def get_tiled_locations(n, metric):
    result = [i * (metric - 1) // n for i in range(n + 1)]
    return result

def behavioral_assay(args, device, actor_critic, exp_name, original_exp_name, num_eval_episodes, num_pos_divisions=4):
    records = []
    for agent_init_pos_x in get_tiled_locations(num_pos_divisions, args.arena_width):
        for agent_init_pos_y in get_tiled_locations(num_pos_divisions, args.arena_height):
            for num_food in [10, 40]:
                for init_energy_level in [0.25, 0.5, 0.75, 1.0]:
                    args.agent_init_pos_mode = 'fixed'
                    args.agent_init_pos_x = agent_init_pos_x
                    args.agent_init_pos_y = agent_init_pos_y
                    args.num_food = num_food
                    args.init_energy_level = init_energy_level

                    env = make_vec_envs(args.env_name,
                                        args.seed,
                                        1,  # num_processes
                                        args.gamma,
                                        args.log_dir,
                                        device,
                                        True,
                                        args
                                        )
                    eval_record = eval_lite(env, args, device, actor_critic, exp_name, num_eval_episodes, make_graphs=False)
                    print(args.agent_init_pos_x, args.agent_init_pos_y, args.num_food, args.init_energy_level)
                    print('eval_record', eval_record, flush=True)

                    record = {
                        'agent_init_pos_x': agent_init_pos_x,
                        'agent_init_pos_y': agent_init_pos_y,
                        'num_food': num_food,
                        'init_energy_level': init_energy_level,
                        'r_mean': eval_record['r_mean'],
                        'r_std': eval_record['r_std'],
                        'steps_mean': eval_record['steps_mean'],
                        't': eval_record['t']
                    }
                    records.append(record)

    df = pd.DataFrame(records)
    save_path = os.path.join(args.log_dir, original_exp_name) #args.save_dir # os.path.join(args.save_dir, args.algo)
    os.makedirs(save_path, exist_ok=True)
    df.to_csv(f'{save_path}/{exp_name}_behavioral_assay.csv')
    print("Summary Statistics:")
    print(df[['r_mean', 'r_std', 'steps_mean', 't']].describe().round(3))
