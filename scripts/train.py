"""Training entry point for Qflex and baseline algorithms."""

import argparse
import os.path
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import envs
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import yaml

import gymnasium

try:
    import myosuite
except ImportError:
    pass

try:
    import humanoid_bench
except ImportError:
    pass

from relax.algorithm.sac import SAC
from relax.algorithm.dsact import DSACT
from relax.algorithm.dacer import DACER
from relax.algorithm.dacer_doubleq import DACERDoubleQ
from relax.algorithm.qsm import QSM
from relax.algorithm.dipo import DIPO
from relax.algorithm.qvpo import QVPO
from relax.algorithm.sdac import SDAC
from relax.algorithm.dpmd import DPMD
from relax.algorithm.idem import IDEM
from relax.algorithm.crossq import CrossQ
from relax.algorithm.flow_exp import FlowExp
from relax.buffer import TreeBuffer
from relax.network.sac import create_sac_net
from relax.network.dsact import create_dsact_net
from relax.network.dacer import create_dacer_net
from relax.network.dacer_doubleq import create_dacer_doubleq_net
from relax.network.qsm import create_qsm_net
from relax.network.dipo import create_dipo_net
from relax.network.diffv2 import create_diffv2_net
from relax.network.qvpo import create_qvpo_net
from relax.network.crossq import create_crossq_net
from relax.network.flow import create_flow_net
from relax.trainer.off_policy import OffPolicyTrainer
from relax.env import create_env, create_vector_env
from relax.utils.experience import Experience, ObsActionPair
from relax.utils.fs import PROJECT_ROOT
from relax.utils.random_utils import seeding
from relax.utils.log_diff import log_git_details


def mish_activation(x: jax.Array):
    """Return the Mish activation used by diffusion-style policy networks."""
    return x * jnp.tanh(jax.nn.softplus(x))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg", type=str, default="sdac")
    parser.add_argument("--env", type=str, default="myoLegWalk-v0")

    parser.add_argument("--suffix", type=str, default="test_use_atp1")
    parser.add_argument("--num_vec_envs", type=int, default=8)
    parser.add_argument("--hidden_num", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--diffusion_steps", type=int, default=20)
    parser.add_argument("--diffusion_hidden_dim", type=int, default=256)
    parser.add_argument("--start_step", type=int, default=int(3e4))
    parser.add_argument("--total_step", type=int, default=int(5e6))
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr_schedule_end", type=float, default=3e-5)
    parser.add_argument("--alpha_lr", type=float, default=3e-4)
    parser.add_argument("--delay_alpha_update", type=float, default=250)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--num_particles", type=int, default=32)
    parser.add_argument("--noise_scale", type=float, default=0.1)
    parser.add_argument("--cluster", default=False, action="store_true")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--beta_schedule_scale", type=float, default=0.8)
    parser.add_argument("--beta_schedule_type", type=str, default="linear")
    parser.add_argument("--grad_step_size", type=float, default=1e-2)
    parser.add_argument("--grad_step_num", type=int, default=20)
    parser.add_argument("--record_video", action="store_true", default=False)
    args = parser.parse_args()

    if args.debug:
        from jax import config

        config.update("jax_disable_jit", True)

    master_seed = args.seed
    master_rng, _ = seeding(master_seed)
    env_seed, env_action_seed, eval_env_seed, buffer_seed, init_network_seed, train_seed = map(
        int, master_rng.integers(0, 2**32 - 1, 6)
    )
    init_network_key = jax.random.key(init_network_seed)
    train_key = jax.random.key(train_seed)
    del init_network_seed, train_seed

    if args.num_vec_envs > 0:
        env, obs_dim, act_dim = create_vector_env(args.env, args.num_vec_envs, env_seed, env_action_seed, mode="futex")
    else:
        env, obs_dim, act_dim = create_env(args.env, env_seed, env_action_seed)
    eval_env = None

    hidden_sizes = [args.hidden_dim] * args.hidden_num
    diffusion_hidden_sizes = [args.diffusion_hidden_dim] * args.hidden_num

    buffer = TreeBuffer.from_experience(obs_dim, act_dim, size=int(1e6), seed=buffer_seed)

    gelu = partial(jax.nn.gelu, approximate=False)
    mish = mish_activation

    print(f"Algorithm: {args.alg}")

    if args.alg == "sdac":
        agent, params = create_diffv2_net(
            init_network_key,
            obs_dim,
            act_dim,
            hidden_sizes,
            diffusion_hidden_sizes,
            mish,
            num_timesteps=args.diffusion_steps,
            num_particles=args.num_particles,
            noise_scale=args.noise_scale,
            beta_schedule_scale=args.beta_schedule_scale,
        )
        algorithm = SDAC(
            agent,
            params,
            lr=args.lr,
            alpha_lr=args.alpha_lr,
            delay_alpha_update=args.delay_alpha_update,
            lr_schedule_end=args.lr_schedule_end,
        )

    elif args.alg == "dpmd":
        agent, params = create_diffv2_net(
            init_network_key,
            obs_dim,
            act_dim,
            hidden_sizes,
            diffusion_hidden_sizes,
            mish,
            num_timesteps=args.diffusion_steps,
            num_particles=args.num_particles,
            noise_scale=args.noise_scale,
            beta_schedule_scale=args.beta_schedule_scale,
        )
        algorithm = DPMD(
            agent,
            params,
            lr=args.lr,
            alpha_lr=args.alpha_lr,
            delay_alpha_update=args.delay_alpha_update,
            lr_schedule_end=args.lr_schedule_end,
        )

    elif args.alg == "idem":
        agent, params = create_diffv2_net(
            init_network_key,
            obs_dim,
            act_dim,
            hidden_sizes,
            diffusion_hidden_sizes,
            mish,
            num_timesteps=args.diffusion_steps,
            num_particles=args.num_particles,
            noise_scale=args.noise_scale,
            beta_schedule_scale=args.beta_schedule_scale,
        )
        algorithm = IDEM(
            agent,
            params,
            lr=args.lr,
            alpha_lr=args.alpha_lr,
            delay_alpha_update=args.delay_alpha_update,
            lr_schedule_end=args.lr_schedule_end,
        )
    elif args.alg == "qsm":
        agent, params = create_qsm_net(
            init_network_key, obs_dim, act_dim, hidden_sizes, num_timesteps=20, num_particles=args.num_particles
        )
        algorithm = QSM(agent, params, lr=args.lr, lr_schedule_end=args.lr_schedule_end)
    elif args.alg == "sac":
        agent, params = create_sac_net(init_network_key, obs_dim, act_dim, hidden_sizes, gelu)
        algorithm = SAC(agent, params, lr=args.lr)
    elif args.alg == "dsact":
        agent, params = create_dsact_net(init_network_key, obs_dim, act_dim, hidden_sizes, gelu)
        algorithm = DSACT(agent, params, lr=args.lr)
    elif args.alg == "dacer":
        agent, params = create_dacer_net(
            init_network_key,
            obs_dim,
            act_dim,
            hidden_sizes,
            diffusion_hidden_sizes,
            mish,
            num_timesteps=args.diffusion_steps,
        )
        algorithm = DACER(agent, params, lr=args.lr, lr_schedule_end=args.lr_schedule_end)
    elif args.alg == "dacer_doubleq":
        agent, params = create_dacer_doubleq_net(
            init_network_key,
            obs_dim,
            act_dim,
            hidden_sizes,
            diffusion_hidden_sizes,
            mish,
            num_timesteps=args.diffusion_steps,
        )
        algorithm = DACERDoubleQ(agent, params, lr=args.lr)
    elif args.alg == "dipo":
        diffusion_buffer = TreeBuffer.from_example(
            ObsActionPair.create_example(obs_dim, act_dim),
            args.total_step,
            int(master_rng.integers(0, 2**32 - 1)),
            remove_batch_dim=False,
        )
        TreeBuffer.connect(buffer, diffusion_buffer, lambda exp: ObsActionPair(exp.obs, exp.action))

        agent, params = create_dipo_net(init_network_key, obs_dim, act_dim, hidden_sizes, num_timesteps=100)
        algorithm = DIPO(
            agent,
            params,
            diffusion_buffer,
            lr=args.lr,
            action_gradient_steps=30,
            policy_target_delay=2,
            action_grad_norm=0.16,
        )
    elif args.alg == "qvpo":
        agent, params = create_qvpo_net(
            init_network_key,
            obs_dim,
            act_dim,
            hidden_sizes,
            diffusion_hidden_sizes,
            mish,
            num_timesteps=args.diffusion_steps,
            num_particles=args.num_particles,
            noise_scale=args.noise_scale,
        )
        algorithm = QVPO(agent, params, lr=args.lr, alpha_lr=args.alpha_lr, delay_alpha_update=args.delay_alpha_update)
    elif args.alg == "crossq":
        agent, params = create_crossq_net(
            init_network_key,
            obs_dim,
            act_dim,
            hidden_sizes,
        )
        algorithm = CrossQ(agent, params, lr=args.lr, alpha_lr=args.alpha_lr)
    elif args.alg == "qflex":
        agent, params = create_flow_net(
            init_network_key,
            obs_dim,
            act_dim,
            hidden_sizes,
            num_timesteps=args.diffusion_steps,
            use_bn=True,
            learn_reference_gn=True,
        )
        algorithm = FlowExp(
            agent,
            params,
            lr=args.lr,
            alpha_lr=args.alpha_lr,
            grad_step_size=args.grad_step_size,
            grad_step_num=args.grad_step_num,
        )
    else:
        raise ValueError(f"Invalid algorithm {args.alg}!")

    if args.cluster:
        PROJECT_ROOT = Path("/n/netscratch/nali_lab_seas/Lab/haitongma/sdac_logs")

    exp_dir = (
        PROJECT_ROOT
        / "logs"
        / args.env
        / (args.alg + "_" + time.strftime("%Y-%m-%d_%H-%M-%S") + f"_s{args.seed}_{args.suffix}")
    )
    trainer = OffPolicyTrainer(
        env=env,
        algorithm=algorithm,
        buffer=buffer,
        start_step=args.start_step,
        total_step=args.total_step,
        sample_per_iteration=1,
        update_per_iteration=1,
        evaluate_env=eval_env,
        evaluate_n_episode=5,
        save_policy_every=min(int(1e6), int(args.total_step / 20)),
        warmup_with="random",
        log_path=exp_dir,
        update_log_n_step=1 if args.debug else 1000,
        record_video=args.record_video,
    )

    trainer.setup(Experience.create_example(obs_dim, act_dim, trainer.batch_size))
    log_git_details(log_file=os.path.join(exp_dir, f"{args.alg}.diff"))

    args_dict = vars(args)
    with open("./relax/algorithm/flow_exp.py", "r") as f:
        flow_code = f.read()
        f.close()
    args_dict["flow_file"] = flow_code
    with open(os.path.join(exp_dir, "config.yaml"), "w") as yaml_file:
        yaml.dump(args_dict, yaml_file)
    trainer.run(train_key)
