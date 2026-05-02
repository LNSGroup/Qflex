import os
from pathlib import Path
import pickle

from typing import NamedTuple, Tuple
import numpy as np
import jax
import jax.numpy as jnp

import optax
import haiku as hk

from relax.algorithm.base import Algorithm
from relax.utils.experience import Experience
from relax.utils.typing_utils import Metric
from relax.network.flow import FlowNet, FlowParams
from relax.utils.persistence import make_persist


class FlowExpOptStates(NamedTuple):
    """Optimizer states for the Qflex trainable components."""

    q1: optax.OptState
    q2: optax.OptState
    velocity_field: optax.OptState
    reference_gn: optax.OptState
    log_alpha: optax.OptState


class FlowExpTrainState(NamedTuple):
    """Full Qflex training state."""

    params: FlowParams
    opt_state: FlowExpOptStates


class FlowExp(Algorithm):
    """Off-policy Qflex algorithm with Q-guided flow construction."""

    def __init__(
        self,
        agent: FlowNet,
        params: FlowParams,
        *,
        gamma: float = 0.99,
        lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        tau: float = 1.0,
        reward_scale: float = 1,
        grad_step_size: float = 1e-2,
        grad_step_num: int = 20,
    ):
        """Initialize Qflex networks, optimizers, and the JIT update."""
        self.agent = agent
        self.gamma = gamma
        self.tau = tau

        self.q_optim = optax.adam(lr, b1=0.5)
        self.velocity_field_optim = optax.adam(lr, b1=0.5)
        self.reference_gn_optim = optax.adam(lr, b1=0.5) if agent.reference_gn is not None else None

        self.log_alpha_optim = optax.adam(alpha_lr)
        self.reward_scale = reward_scale
        self.grad_step_size = grad_step_size
        self.grad_step_num = grad_step_num
        self.updated_policy_structure = False
        self.state = FlowExpTrainState(
            params=params,
            opt_state=FlowExpOptStates(
                q1=self.q_optim.init(params.q1),
                q2=self.q_optim.init(params.q2),
                velocity_field=self.velocity_field_optim.init(params.velocity_field),
                reference_gn=(
                    self.reference_gn_optim.init(params.reference_gn) if self.reference_gn_optim is not None else None
                ),
                log_alpha=self.log_alpha_optim.init(params.log_alpha),
            ),
        )
        self.first_update = True

        @jax.jit
        def stateless_update(
            key: jax.Array, state: FlowExpTrainState, data: Experience
        ) -> Tuple[FlowExpTrainState, Metric]:
            """Run one Qflex update step without mutating object state."""
            obs, action, reward, next_obs, done = data.obs, data.action, data.reward, data.next_obs, data.done
            obs = jnp.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
            next_obs = jnp.nan_to_num(next_obs, nan=0.0, posinf=0.0, neginf=0.0)
            action = jnp.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
            reward = jnp.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)

            (
                q1_params,
                q2_params,
                target_q1_params,
                target_q2_params,
                velocity_field_params,
                reference_gn_params,
                q1_state,
                q2_state,
                velocity_field_state,
                reference_gn_state,
                log_alpha,
                exp_prob,
            ) = state.params
            q1_opt_state, q2_opt_state, velocity_field_opt_state, reference_gn_opt_state, log_alpha_opt_state = (
                state.opt_state
            )
            ref_eval_key, vel_eval_key, mc_normal_eval_key = jax.random.split(key, 3)

            reward *= self.reward_scale

            next_action, next_logp, _, _, _ = self.agent.evaluate_reference(
                ref_eval_key, reference_gn_params, reference_gn_state, next_obs, is_training=False
            )

            action_max = jnp.max(jnp.abs(action), axis=1, keepdims=True)
            next_logp_sum = jnp.sum(next_logp, axis=-1)
            catted_q1, _ = self.agent.q(
                q1_params,
                q1_state,
                jnp.concatenate([obs, next_obs], axis=0),
                jnp.concatenate([action, next_action], axis=0),
                is_training=True,
            )

            catted_q2, _ = self.agent.q(
                q2_params,
                q2_state,
                jnp.concatenate([obs, next_obs], axis=0),
                jnp.concatenate([action, next_action], axis=0),
                is_training=True,
            )

            _, q1_target = jnp.split(catted_q1, 2)
            _, q2_target = jnp.split(catted_q2, 2)

            q_target = jnp.minimum(q1_target, q2_target)
            q_backup = reward + (1 - done) * self.gamma * q_target

            def q_loss_fn(q_params: hk.Params, q_state: hk.State) -> jax.Array:
                """Compute Bellman regression loss for one Q network."""
                catted_q, q_state = self.agent.q(
                    q_params,
                    q_state,
                    jnp.concatenate([obs, next_obs], axis=0),
                    jnp.concatenate([action, next_action], axis=0),
                    is_training=True,
                )
                current_q_values, _ = jnp.split(catted_q, 2)

                q_loss = jnp.mean((current_q_values - jax.lax.stop_gradient(q_backup)) ** 2)
                return q_loss, q_state

            (q1_loss, q1_state), q1_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(q1_params, q1_state)
            (q2_loss, q2_state), q2_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(q2_params, q2_state)

            q1_update, q1_opt_state = self.q_optim.update(q1_grads, q1_opt_state)
            q2_update, q2_opt_state = self.q_optim.update(q2_grads, q2_opt_state)
            q1_params = optax.apply_updates(q1_params, q1_update)
            q2_params = optax.apply_updates(q2_params, q2_update)

            def reference_gn_loss_fn(
                reference_gn_params: hk.Params,
                reference_gn_state: hk.State,
                q1_state: hk.State,
                q2_state: hk.State,
            ) -> jax.Array:
                """Train the reference Gaussian toward high-Q actions."""
                new_action, new_logp, reference_gn_state, new_mean, new_std = self.agent.evaluate_reference(
                    ref_eval_key, reference_gn_params, reference_gn_state, obs, is_training=True
                )

                q1, _ = self.agent.q(target_q1_params, q1_state, obs, new_action, is_training=False)
                q2, _ = self.agent.q(target_q2_params, q2_state, obs, new_action, is_training=False)
                q = jnp.minimum(q1, q2)

                reference_gn_loss = -q.reshape(-1, 1)
                return jnp.mean(reference_gn_loss), (q1, q2, reference_gn_state, new_logp)

            if self.agent.reference_gn is not None:
                (reference_gn_loss, aux), reference_gn_grads = jax.value_and_grad(reference_gn_loss_fn, has_aux=True)(
                    reference_gn_params,
                    reference_gn_state,
                    q1_state,
                    q2_state,
                )
                q1, q2, reference_gn_state, new_logp = aux

                reference_gn_update, reference_gn_opt_state = self.reference_gn_optim.update(
                    reference_gn_grads, reference_gn_opt_state
                )
                reference_gn_params = optax.apply_updates(reference_gn_params, reference_gn_update)
            else:
                reference_gn_loss = 0
                new_logp = 0

            def velocity_field_loss_fn(
                velocity_field_params: hk.Params,
                reference_gn_params: hk.Params,
                velocity_field_state: hk.State,
                reference_gn_state: hk.State,
            ) -> jax.Array:
                """Fit the velocity field to the Q-guided action update."""

                grad_step_num = self.grad_step_num

                action_init, init_logp, reference_gn_state, reference_mean, reference_std = (
                    self.agent.evaluate_reference(
                        ref_eval_key, reference_gn_params, reference_gn_state, next_obs, is_training=False
                    )
                )

                def q_eval(action):
                    """Evaluate the clipped double-Q target for a batch of actions."""
                    q1, _ = self.agent.q(target_q1_params, q1_state, next_obs, action, is_training=False)
                    q2, _ = self.agent.q(target_q2_params, q2_state, next_obs, action, is_training=False)
                    q = jnp.minimum(q1, q2)
                    return q

                def q_singe_eval(single_obs, single_action):
                    """Evaluate one observation/action pair for action gradients."""
                    single_obs = single_obs.reshape(1, -1)
                    single_action = single_action.reshape(1, -1)
                    q1, _ = self.agent.q(target_q1_params, q1_state, single_obs, single_action, is_training=False)
                    q2, _ = self.agent.q(target_q2_params, q2_state, single_obs, single_action, is_training=False)
                    q = jnp.minimum(q1, q2)
                    return q

                q_init = q_eval(action_init)
                lb = -jnp.ones_like(action_init)
                ub = jnp.ones_like(action_init)

                def q_flow_eval(x, single_obs):
                    """Scalar Q target used by ``jax.grad`` over the action."""
                    q = q_singe_eval(single_obs, x)
                    return q[0]

                grad_step_size = self.grad_step_size

                def q_flow_construction(y, noise):
                    """Apply one bounded Q-gradient ascent step in action space."""
                    y_init = y

                    grad_y = jax.grad(q_flow_eval, argnums=0)
                    batched_grad_q = jax.vmap(grad_y)
                    q_grad_init = batched_grad_q(y_init, next_obs)

                    max_update = 2 * jnp.sqrt(y.shape[-1])
                    grad_norm = jnp.linalg.norm(q_grad_init, axis=1, keepdims=True)
                    grad_step_size_vec = grad_step_size * jnp.ones_like(grad_norm)
                    # Bound the update by action dimension to avoid exploding flow targets.
                    grad_step_size_vec = jnp.minimum(grad_step_size_vec, max_update / (grad_norm + 1e-6))

                    y_init += grad_step_size_vec * q_grad_init
                    return y_init, (y_init, jnp.linalg.norm(q_grad_init), jnp.max(jnp.abs(q_grad_init)))

                noises = jax.random.normal(mc_normal_eval_key, (grad_step_num, action.shape[0], action.shape[1]))

                final_update, updated_init = jax.lax.scan(q_flow_construction, action_init, noises)

                grad_norm = jnp.mean(updated_init[1])
                grad_max = jnp.mean(updated_init[2])

                y_flow_update = final_update
                y_diff = y_flow_update - action_init
                action_flow_update = y_flow_update

                action_flow_update = jax.lax.stop_gradient(action_flow_update)

                q_flow_update = q_eval(action_flow_update)

                q_diff_init = q_flow_update - q_init
                q_diff_flow = q_diff_init

                mc_num = 1
                wide_obs = jnp.repeat(next_obs, mc_num, axis=0)
                t = jax.random.uniform(vel_eval_key, (next_obs.shape[0] * mc_num, 1))

                normal_sample = action_init
                wide_normal_sample = jnp.repeat(normal_sample, mc_num, axis=0)

                wide_updated_action = jnp.repeat(action_flow_update, mc_num, axis=0)
                temp_action = (1 - t) * wide_normal_sample + t * wide_updated_action
                temp_velocity, velocity_field_state = self.agent.flow.velocity_field_apply(
                    velocity_field_params, velocity_field_state, wide_obs, temp_action, t, is_training=True
                )

                target_velocity = wide_updated_action - wide_normal_sample
                velocity_loss = (temp_velocity - target_velocity) ** 2

                target_vel_norm = jnp.linalg.norm(target_velocity, axis=1)
                target_vel_max = jnp.max(jnp.abs(target_velocity), axis=1)

                bound__width = ub - lb
                return jnp.mean(velocity_loss), (
                    velocity_field_state,
                    target_vel_norm,
                    q_flow_update,
                    q_init,
                    q_diff_init,
                    q_diff_flow,
                    y_diff,
                    target_vel_max,
                    grad_norm,
                    grad_max,
                    bound__width,
                )

            (velocity_field_loss, aux), velocity_field_grads = jax.value_and_grad(velocity_field_loss_fn, has_aux=True)(
                velocity_field_params,
                reference_gn_params,
                velocity_field_state,
                reference_gn_state,
            )
            (
                velocity_field_state,
                target_vel_norm,
                q_flow_update,
                q_init,
                q_diff_init,
                q_diff_flow,
                y_diff,
                target_vel_max,
                grad_norm,
                grad_max,
                bound_width,
            ) = aux
            velocity_field_update, velocity_field_opt_state = self.velocity_field_optim.update(
                velocity_field_grads, velocity_field_opt_state
            )
            velocity_field_params = optax.apply_updates(velocity_field_params, velocity_field_update)

            action_flow, action_init = self.agent.flow.sample(
                key,
                obs,
                velocity_field_params,
                velocity_field_state,
                reference_gn_params,
                reference_gn_state,
                is_training=False,
            )

            def q_eval(action, obs):
                q1, _ = self.agent.q(target_q1_params, q1_state, obs, action, is_training=False)
                q2, _ = self.agent.q(target_q2_params, q2_state, obs, action, is_training=False)
                q = jnp.minimum(q1, q2)
                return q

            q_exp_flow = q_eval(action_flow, obs)
            q_exp_init = q_eval(action_init, obs)
            max_idx = q_exp_flow > q_exp_init

            exp_prob = 1

            target_q1_params = optax.incremental_update(q1_params, target_q1_params, self.tau)
            target_q2_params = optax.incremental_update(q2_params, target_q2_params, self.tau)

            state = FlowExpTrainState(
                params=FlowParams(
                    q1_params,
                    q2_params,
                    target_q1_params,
                    target_q2_params,
                    velocity_field_params,
                    reference_gn_params,
                    q1_state,
                    q2_state,
                    velocity_field_state,
                    reference_gn_state,
                    log_alpha,
                    exp_prob,
                ),
                opt_state=FlowExpOptStates(
                    q1_opt_state, q2_opt_state, velocity_field_opt_state, reference_gn_opt_state, log_alpha_opt_state
                ),
            )

            info = {
                "q1_loss": q1_loss,
                "q2_loss": q2_loss,
                "q_flow_update": jnp.mean(q_flow_update),
                "q_init": jnp.mean(q_init),
                "reference_gn_loss": reference_gn_loss,
                "velocity_field_loss": velocity_field_loss,
                "target_velocity_norm": jnp.mean(target_vel_norm),
                "entropy": -jnp.mean(next_logp_sum),
                "entropy_min": -jnp.max(next_logp_sum),
                "reference_entropy": -jnp.mean(jnp.sum(new_logp, axis=-1)),
                "max_idx": jnp.mean(max_idx),
                "max_action": jnp.mean(action_max),
                "exp_mode": exp_prob,
                "grad_step_diff_init": jnp.mean(q_diff_init),
                "grad_step_diff_flow": jnp.mean(q_diff_flow),
                "bijection_diff": jnp.mean(jnp.linalg.norm(y_diff, axis=1)),
                "bijection_max": jnp.mean(jnp.max(jnp.abs(y_diff), axis=1)),
                "target_velocity_max": jnp.mean(target_vel_max),
                "grad_step_grad_norm": grad_norm,
                "grad_step_grad_max": grad_max,
                "bound_width_mean": jnp.mean(jnp.mean(bound_width, axis=1)),
                "bound_width_max": jnp.mean(jnp.max(bound_width, axis=1)),
            }

            return state, info

        self._implement_common_behavior(stateless_update, self.agent.get_action, self.agent.get_deterministic_action)

    def get_policy_params(self):
        """Return policy parameters used by action sampling."""
        return (self.state.params.velocity_field, self.state.params.reference_gn, self.state.params.exp_prob)

    def get_policy_state(self):
        """Return policy Haiku state used by action sampling."""
        return (self.state.params.velocity_field_state, self.state.params.reference_gn_state)

    def warmup(self, data: Experience) -> None:
        """Compile update and action functions with representative data."""
        key = jax.random.key(0)
        obs = data.obs[0]
        self.input_shape = obs.shape
        policy_params = self.get_policy_params()
        policy_state = self.get_policy_state()
        self._update(key, self.state, data)
        self._get_action(key, policy_params, policy_state, obs)
        self._get_deterministic_action(policy_params, policy_state, obs)

    def get_action(self, key: jax.Array, obs: np.ndarray) -> np.ndarray:
        """Sample a stochastic action as a NumPy array."""
        action = self._get_action(key, self.get_policy_params(), self.get_policy_state(), obs)
        return np.asarray(action)

    def get_deterministic_action(self, obs: np.ndarray) -> np.ndarray:
        """Sample the deterministic evaluation action as a NumPy array."""
        key = jnp.random.key(0)
        action = self._get_action(key, self.get_policy_params(), self.get_policy_state(), obs)
        return np.asarray(action)

    def save_policy_structure(self, root: os.PathLike, dummy_obs: jax.Array) -> None:
        """Persist traced stochastic and deterministic policy callables."""
        self.root_path = root
        root = Path(root)
        policy_state = jax.device_get(self.get_policy_state())
        key = jax.random.key(0)
        stochastic = make_persist(self._get_action._fun)(key, self.get_policy_params(), policy_state, dummy_obs)
        deterministic = make_persist(self._get_deterministic_action._fun)(
            self.get_policy_params(), policy_state, dummy_obs
        )

        stochastic.save(root / "stochastic.pkl")
        stochastic.save_info(root / "stochastic.txt")
        deterministic.save(root / "deterministic.pkl")
        deterministic.save_info(root / "deterministic.txt")

    def save_policy(self, path: str) -> None:
        """Save policy parameters and state to disk."""
        policy = jax.device_get(self.get_policy_params())
        policy_state = jax.device_get(self.get_policy_state())
        with open(path, "wb") as f:
            pickle.dump((policy, policy_state), f)
