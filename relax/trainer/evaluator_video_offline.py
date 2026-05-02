import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
from functools import partial

from pathlib import Path
import argparse
import pickle
import csv
import yaml
import numpy as np
import jax
import mujoco
import jax.numpy as jnp
from tensorboardX import SummaryWriter

from relax.env import create_env
from relax.utils.persistence import PersistFunction
from relax.network.crossq import create_crossq_net
from relax.network.flow import create_flow_net
from gymnasium.wrappers import RecordVideo
try:
    from myosuite.utils import gym
except:
    pass # not use myosuite, just ignore the error
try:
    import humanoid_bench
except:
    pass # not use humanoid_bench, just ignore the error

def stop_video_recording(env):
    if hasattr(env, "stop_recording"):
        if getattr(env, "recording", False):
            env.stop_recording()
    elif hasattr(env, "close_video_recorder"):
        env.close_video_recorder()

def evaluate(env, policy_fn, policy_params, num_episodes, step):
    ep_len_list = []
    ep_ret_list = []

    env = RecordVideo(env, f"{args.policy_root}/{step}_eval", episode_trigger=lambda x: True, disable_logger=True)
    if hasattr(env, "start_video_recorder"):
        env.start_video_recorder()
    for ep_num in range(num_episodes):
        obs, _ = env.reset()
        ep_len = 0
        ep_ret = 0.0
        ep_info = {}
        all_qpos = env.data.qpos.copy().reshape(1, -1)
        all_act = env.data.act.copy().reshape(1, -1)
        while True:
            act = policy_fn(policy_params, obs)
            act = np.array(act)
            obs, reward, terminated, truncated, info = env.step(act)
            for key in info:
                if key not in ep_info:
                    ep_info[key] = []
                ep_info[key].append(info[key])
            ep_len += 1
            ep_ret += reward
            all_qpos = np.concatenate([all_qpos, env.data.qpos.copy().reshape(1, -1)], axis=0)
            all_act = np.concatenate([all_act, env.data.act.copy().reshape(1, -1)], axis=0)
            if terminated or truncated:
                break
        ep_len_list.append(ep_len)
        ep_ret_list.append(ep_ret)
        ep_mean_info = {key: np.mean(ep_info[key]) for key in ep_info}
        # np.savez(f"{args.policy_root}/{step}_eval/{ep_num}_qpos.npz", qpos=all_qpos)
        np.save(f"{args.policy_root}/{step}_eval/{ep_num}_qpos.npy", all_qpos)
        np.save(f"{args.policy_root}/{step}_eval/{ep_num}_act.npy", all_act)

    stop_video_recording(env)

    
    print('finish eval')
    return ep_len_list, ep_ret_list, ep_mean_info

class Logger(object):

    def __init__(self, log_dir, info):
        self.path = os.path.join(log_dir, 'log.csv')
        row = ['step', 'avg_ret', 'std_ret']
        for key in info:
            row.append(key)
        with open(self.path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def log(self, step, avg_ret, std_ret, info):
        row = [step, avg_ret, std_ret]
        for key in info:
            row.append(info[key])
        with open(self.path, mode='a', newline='') as f:
            writer = csv.writer(f)
            
            writer.writerow(row)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("policy_root", type=Path)
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--num_episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    master_rng = np.random.default_rng(args.seed)
    env_seed, env_action_seed, policy_seed = map(int, master_rng.integers(0, 2**32 - 1, 3))
    env, _, _ = create_env(args.env, env_seed, env_action_seed)
    try:
        env.render()
        env.mujoco_renderer.viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR]=True
        env.mujoco_renderer.viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_ACTIVATION]=True
    except Exception as e:
        # print(e)
        # raise NotImplementedError
        pass
    obs, _ = env.reset()
    _, _, _, _, info = env.step(env.action_space.sample())
    print(args.policy_root)
    if 'crossq' in args.policy_root.name:
         # read config file
        with open(f'{args.policy_root}/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        hidden_num = config['hidden_num']
        hidden_dim = config['hidden_dim']
        hidden_sizes = [hidden_dim] * hidden_num
        policy_key = jax.random.key(policy_seed)
        def mish(x: jax.Array):
            return x * jnp.tanh(jax.nn.softplus(x))
        agent, init_params = create_crossq_net(policy_key, obs_dim=env.observation_space.shape[0], act_dim=env.action_space.shape[0], hidden_sizes=hidden_sizes, 
                                            #    activation=mish
        )
    elif 'flow' in args.policy_root.name:
        with open(f'{args.policy_root}/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        hidden_num = config['hidden_num']
        hidden_dim = config['hidden_dim']
        hidden_sizes = [hidden_dim] * hidden_num
        policy_key = jax.random.key(policy_seed)
        def mish(x: jax.Array):
            return x * jnp.tanh(jax.nn.softplus(x))
        agent, init_params = create_flow_net(policy_key, obs_dim=env.observation_space.shape[0], act_dim=env.action_space.shape[0], hidden_sizes=hidden_sizes, 
                                            #    activation=mish
        )
    
    policy = PersistFunction.load(args.policy_root / "deterministic.pkl")
    @jax.jit
    def policy_fn(policy_params, obs):
        if 'crossq' in args.policy_root.name or 'flow' in args.policy_root.name:
            policy_param, policy_state = policy_params
            action = agent.get_deterministic_action(policy_param, policy_state, obs).clip(-1, 1)
            # print('crossq action', action)
            return action.flatten()
        else:
            return policy(policy_params, obs).clip(-1, 1)
            
    # logger = SummaryWriter(args.policy_root)
    # logger = Logger(args.policy_root, info)

    while payload := sys.stdin.readline():
        step, policy_path = payload.strip().split(",", maxsplit=1)
        step = int(step)
        with open(policy_path, "rb") as f:
            policy_params = pickle.load(f)

        ep_len_list, ep_ret_list, ep_mean_info = evaluate(env, policy_fn, policy_params, args.num_episodes, step)

        # ep_len = np.array(ep_len_list)
        # ep_ret = np.array(ep_ret_list)
        
        # logger.add_scalar("evaluate/episode_length", ep_len_mean.mean(), step)
        # logger.add_scalar("evaluate/episode_return", ep_ret_mean.mean(), step)
        # # logger.add_histogram("evaluate/episode_length", ep_len_mean, step)
        # # logger.add_histogram("evaluate/episode_return", ep_ret_mean, step)
        # logger.flush()
        # logger.log(step, ep_ret.mean(), ep_ret.std(), ep_mean_info)
