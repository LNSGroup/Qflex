import os
import argparse
import csv
import pickle
import sys
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["OMP_NUM_THREADS"] = "1"

import yaml  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from relax.env import create_env  # noqa: E402
from relax.network.crossq import create_crossq_net  # noqa: E402
from relax.network.flow import create_flow_net  # noqa: E402
from relax.utils.persistence import PersistFunction  # noqa: E402


def evaluate(env, policy_fn, policy_params, num_episodes):
    """Evaluate a policy for a fixed number of episodes."""
    ep_len_list = []
    ep_ret_list = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        ep_len = 0
        ep_ret = 0.0
        while True:
            act = policy_fn(policy_params, obs)
            act = np.array(act)
            obs, reward, terminated, truncated, _ = env.step(act)
            ep_len += 1
            ep_ret += reward
            if terminated or truncated:
                break
        ep_len_list.append(ep_len)
        ep_ret_list.append(ep_ret)
    return ep_len_list, ep_ret_list


class Logger:
    """CSV logger for evaluator returns."""

    def __init__(self, log_dir):
        """Create a fresh evaluation log file."""
        self.path = os.path.join(log_dir, "log.csv")
        with open(self.path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "avg_ret", "std_ret"])

    def log(self, step, avg_ret, std_ret):
        """Append one evaluation row."""
        with open(self.path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([step, avg_ret, std_ret])


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
    print(args.policy_root)
    if "crossq" in args.policy_root.name:
        with open(f"{args.policy_root}/config.yaml", "r") as f:
            config = yaml.safe_load(f)
        hidden_num = config["hidden_num"]
        hidden_dim = config["hidden_dim"]
        hidden_sizes = [hidden_dim] * hidden_num
        policy_key = jax.random.key(policy_seed)

        def mish(x: jax.Array):
            """Mish activation used by some policy configs."""
            return x * jnp.tanh(jax.nn.softplus(x))

        agent, init_params = create_crossq_net(
            policy_key,
            obs_dim=env.observation_space.shape[0],
            act_dim=env.action_space.shape[0],
            hidden_sizes=hidden_sizes,
        )
    elif "flow" in args.policy_root.name or "qflex" in args.policy_root.name:
        with open(f"{args.policy_root}/config.yaml", "r") as f:
            config = yaml.safe_load(f)
        hidden_num = config["hidden_num"]
        hidden_dim = config["hidden_dim"]
        hidden_sizes = [hidden_dim] * hidden_num
        policy_key = jax.random.key(policy_seed)

        def mish(x: jax.Array):
            """Mish activation used by some policy configs."""
            return x * jnp.tanh(jax.nn.softplus(x))

        agent, init_params = create_flow_net(
            policy_key,
            obs_dim=env.observation_space.shape[0],
            act_dim=env.action_space.shape[0],
            hidden_sizes=hidden_sizes,
        )

    policy = PersistFunction.load(args.policy_root / "deterministic.pkl")

    @jax.jit
    def policy_fn(policy_params, obs):
        """Dispatch persisted policies and Qflex/CrossQ network policies."""
        if "crossq" in args.policy_root.name or "flow" in args.policy_root.name or "qflex" in args.policy_root.name:
            policy_param, policy_state = policy_params
            action = agent.get_deterministic_action(policy_param, policy_state, obs).clip(-1, 1)
            return action.flatten()
        else:
            return policy(policy_params, obs).clip(-1, 1)

    logger = Logger(args.policy_root)

    while payload := sys.stdin.readline():
        step, policy_path = payload.strip().split(",", maxsplit=1)
        step = int(step)
        with open(policy_path, "rb") as f:
            policy_params = pickle.load(f)

        ep_len_list, ep_ret_list = evaluate(env, policy_fn, policy_params, args.num_episodes)

        ep_len = np.array(ep_len_list)
        ep_ret = np.array(ep_ret_list)
        logger.log(step, ep_ret.mean(), ep_ret.std())
