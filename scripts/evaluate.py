"""Evaluate a saved policy checkpoint and optionally record videos."""

import argparse
import csv
import os
import pickle
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["OMP_NUM_THREADS"] = "1"

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from gymnasium.wrappers import RecordVideo  # noqa: E402

try:
    import myosuite  # noqa: F401, E402
except ImportError:
    pass

try:
    import humanoid_bench  # noqa: F401, E402
except ImportError:
    pass

from relax.env import create_env  # noqa: E402
from relax.network.crossq import create_crossq_net  # noqa: E402
from relax.network.flow import create_flow_net  # noqa: E402
from relax.utils.persistence import PersistFunction  # noqa: E402


NETWORK_POLICY_ALGS = {"crossq", "flow", "flowexp", "qflex"}


def parse_args():
    """Parse command-line arguments for direct checkpoint evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path, help="Path to a saved policy-*.pkl checkpoint.")
    parser.add_argument("--env", type=str, default=None, help="Environment name. Defaults to config.yaml.")
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_video", action="store_true", help="Disable video recording.")
    parser.add_argument("--output_dir", type=Path, default=None, help="Override the evaluation output directory.")
    return parser.parse_args()


def safe_name(name: str) -> str:
    """Return a filesystem-friendly name segment."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def load_config(policy_root: Path) -> dict:
    """Load the training config next to the saved checkpoint."""
    config_path = policy_root / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing training config: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def infer_algorithm(config: dict, policy_root: Path) -> str:
    """Infer the algorithm from config.yaml, falling back to the run folder name."""
    alg = str(config.get("alg", "")).lower()
    if alg:
        return alg
    run_name = policy_root.name.lower()
    for candidate in NETWORK_POLICY_ALGS:
        if candidate in run_name:
            return candidate
    return run_name.split("_", maxsplit=1)[0]


def make_output_dir(model_path: Path, output_dir: Optional[Path]) -> Path:
    """Create the timestamped evaluation directory beside the model file."""
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=False)
        return output_dir

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    eval_name = f"eval_{timestamp}_{safe_name(model_path.stem)}"
    path = model_path.parent / eval_name
    path.mkdir(parents=True, exist_ok=False)
    return path


def stop_video_recording(env):
    """Flush a RecordVideo wrapper across Gymnasium versions."""
    if hasattr(env, "stop_recording"):
        if getattr(env, "recording", False):
            env.stop_recording()
    elif hasattr(env, "close_video_recorder"):
        env.close_video_recorder()


def maybe_enable_mujoco_flags(env):
    """Enable actuator visualization when the MuJoCo viewer exposes it."""
    try:
        env.render()
        env.mujoco_renderer.viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True
        env.mujoco_renderer.viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_ACTIVATION] = True
    except Exception:
        pass


def create_policy_fn(policy_root: Path, config: dict, alg: str, obs_dim: int, act_dim: int, policy_seed: int):
    """Create the deterministic policy function used for evaluation."""
    deterministic_path = policy_root / "deterministic.pkl"
    if not deterministic_path.exists():
        raise FileNotFoundError(f"Missing deterministic policy structure: {deterministic_path}")

    policy = PersistFunction.load(deterministic_path)
    if alg not in NETWORK_POLICY_ALGS:
        return jax.jit(lambda policy_params, obs: policy(policy_params, obs).clip(-1, 1))

    hidden_num = int(config["hidden_num"])
    hidden_dim = int(config["hidden_dim"])
    hidden_sizes = [hidden_dim] * hidden_num
    policy_key = jax.random.key(policy_seed)

    if alg == "crossq":
        agent, _ = create_crossq_net(policy_key, obs_dim=obs_dim, act_dim=act_dim, hidden_sizes=hidden_sizes)
    else:
        diffusion_steps = int(config.get("diffusion_steps", 20))
        agent, _ = create_flow_net(
            policy_key,
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_sizes=hidden_sizes,
            num_timesteps=diffusion_steps,
            use_bn=True,
            learn_reference_gn=True,
        )

    @jax.jit
    def policy_fn(policy_params, obs):
        policy_param, policy_state = policy_params
        action = agent.get_deterministic_action(policy_param, policy_state, obs).clip(-1, 1)
        return action.flatten()

    return policy_fn


def evaluate(env, policy_fn, policy_params, num_episodes: int):
    """Run evaluation episodes and collect returns, lengths, and info metrics."""
    rows = []
    info_values = {}
    for episode in range(num_episodes):
        obs, _ = env.reset()
        ep_len = 0
        ep_ret = 0.0
        ep_info = {}

        while True:
            action = np.array(policy_fn(policy_params, obs))
            obs, reward, terminated, truncated, info = env.step(action)
            if "rwd_dict" in info:
                info = info["rwd_dict"]
            for key, value in info.items():
                ep_info.setdefault(key, []).append(value)
            ep_len += 1
            ep_ret += reward
            if terminated or truncated:
                break

        mean_info = {key: float(np.mean(values)) for key, values in ep_info.items()}
        for key, value in mean_info.items():
            info_values.setdefault(key, []).append(value)
        rows.append({"episode": episode, "return": float(ep_ret), "length": ep_len, **mean_info})

    summary = {
        "avg_return": float(np.mean([row["return"] for row in rows])),
        "std_return": float(np.std([row["return"] for row in rows])),
        "avg_length": float(np.mean([row["length"] for row in rows])),
        "std_length": float(np.std([row["length"] for row in rows])),
    }
    summary.update({key: float(np.mean(values)) for key, values in info_values.items()})
    return rows, summary


def write_metrics(output_dir: Path, rows: list, summary: dict):
    """Write per-episode metrics and aggregate summary files."""
    fieldnames = sorted({key for row in rows for key in row.keys()})
    preferred = ["episode", "return", "length"]
    fieldnames = preferred + [key for key in fieldnames if key not in preferred]
    with open(output_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with open(output_dir / "summary.yaml", "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)


def main():
    """Evaluate one saved model checkpoint from a single command."""
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {model_path}")
    if model_path.name in {"deterministic.pkl", "stochastic.pkl"}:
        raise ValueError("Pass a saved policy parameter file such as policy-1000000-12500.pkl.")

    policy_root = model_path.parent
    config = load_config(policy_root)
    env_name = args.env or config.get("env")
    if not env_name:
        raise ValueError("Environment name is missing. Pass --env or keep env in config.yaml.")

    master_rng = np.random.default_rng(args.seed)
    env_seed, env_action_seed, policy_seed = map(int, master_rng.integers(0, 2**32 - 1, 3))
    env, obs_dim, act_dim = create_env(env_name, env_seed, env_action_seed)

    output_dir = make_output_dir(model_path, args.output_dir)
    if not args.no_video:
        maybe_enable_mujoco_flags(env)
        env = RecordVideo(env, str(output_dir / "videos"), episode_trigger=lambda _: True, disable_logger=True)
        if hasattr(env, "start_video_recorder"):
            env.start_video_recorder()

    alg = infer_algorithm(config, policy_root)
    policy_fn = create_policy_fn(policy_root, config, alg, obs_dim, act_dim, policy_seed)
    with open(model_path, "rb") as f:
        policy_params = pickle.load(f)

    rows, summary = evaluate(env, policy_fn, policy_params, args.num_episodes)
    if not args.no_video:
        stop_video_recording(env)
    env.close()

    summary.update(
        {
            "env": env_name,
            "algorithm": alg,
            "model_path": str(model_path),
            "output_dir": str(output_dir),
            "num_episodes": args.num_episodes,
            "seed": args.seed,
            "record_video": not args.no_video,
        }
    )
    write_metrics(output_dir, rows, summary)

    print(f"Evaluation written to: {output_dir}")
    print(f"Average return: {summary['avg_return']:.3f} +/- {summary['std_return']:.3f}")


if __name__ == "__main__":
    main()
