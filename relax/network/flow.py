from dataclasses import dataclass
from typing import Callable, NamedTuple, Optional, Sequence, Tuple, Union

import haiku as hk
import jax
import jax.numpy as jnp
from numpyro.distributions import Normal

from relax.network.blocks import (
    Activation,
    FlowVelocityFieldBN,
    PolicyNet,
    PolicyNetBN,
    QNetBN,
)
from relax.utils.flow_matching import FlowMatching, simple_euler


class FlowParams(NamedTuple):
    """Trainable parameters and Haiku state for Qflex."""

    q1: hk.Params
    q2: hk.Params
    target_q1: hk.Params
    target_q2: hk.Params
    velocity_field: hk.Params
    reference_gn: hk.Params
    q1_state: hk.State
    q2_state: hk.State
    velocity_field_state: hk.State
    reference_gn_state: hk.Params
    log_alpha: jax.Array
    exp_prob: float


@dataclass
class FlowNet:
    """Network bundle used by the Qflex algorithm."""

    velocity_field: Callable[[hk.Params, jax.Array], Tuple[jax.Array, jax.Array]]
    num_timesteps: int
    act_dim: int
    q: Callable[[hk.Params, jax.Array, jax.Array], jax.Array]
    target_entropy: float
    bn: bool = True
    reference_gn: Optional[Union[PolicyNetBN, PolicyNet]] = None

    @property
    def flow(self) -> FlowMatching:
        """Create the flow-matching helper around the current apply functions."""
        return FlowMatching(
            velocity_field_apply=self.velocity_field.apply,
            act_dim=self.act_dim,
            num_timesteps=self.num_timesteps,
            odeint=simple_euler,
            reference_gn=self.reference_gn,
            use_bn=self.bn,
        )

    def get_action(
        self,
        key: jax.Array,
        policy_params: Union[hk.Params, Tuple[hk.Params, hk.Params]],
        state: Union[hk.State, Tuple[hk.State, hk.State]],
        obs: jax.Array,
    ) -> jax.Array:
        """Sample stochastic actions for environment interaction."""
        if self.bn:
            velocity_field_params, reference_gn_params, exp_prob = policy_params
            velocity_field_state, reference_gn_state = state
        else:
            velocity_field_params, exp_prob = policy_params
            velocity_field_state = state
            reference_gn_params = None
            reference_gn_state = None
        sample_key = jax.random.split(key, 1)[0]
        action, action_init = self.flow.sample(
            sample_key,
            obs,
            velocity_field_params,
            velocity_field_state,
            reference_gn_params,
            reference_gn_state,
            is_training=False,
        )

        exp_key = jax.random.split(key, 1)[0]
        # Mix the learned flow and reference Gaussian according to exp_prob.
        exp_mode = jax.random.choice(
            exp_key,
            jnp.array([0, 1]),
            p=jnp.array([1 - exp_prob, exp_prob]),
            shape=(action.shape[0], 1),
        )
        action = exp_mode * action + (1 - exp_mode) * action_init

        return action

    def get_deterministic_action(
        self,
        policy_params: Union[hk.Params, Tuple[hk.Params, hk.Params]],
        state: Union[hk.State, Tuple[hk.State, hk.State]],
        obs: jax.Array,
    ) -> jax.Array:
        """Return the flow action used for deterministic evaluation."""
        if self.bn:
            velocity_field_params, reference_gn_params, _ = policy_params
            velocity_field_state, reference_gn_state = state
        else:
            velocity_field_params = policy_params
            velocity_field_state = state
            reference_gn_params = None
            reference_gn_state = None
        key = jax.random.key(0)
        action, action_init = self.flow.sample(
            key,
            obs,
            velocity_field_params,
            velocity_field_state,
            reference_gn_params,
            reference_gn_state,
            is_training=False,
        )
        return action

    def evaluate_reference(
        self,
        key: jax.Array,
        reference_gn_params: hk.Params,
        reference_gn_state: hk.State,
        obs: jax.Array,
        is_training: bool = True,
    ) -> Tuple[jax.Array, jax.Array]:
        """Sample from the learned reference Gaussian for algorithm updates."""
        (mean, std), state = self.flow.reference_gn.apply(
            reference_gn_params, reference_gn_state, obs, is_training=is_training
        )
        z = jax.random.normal(key, mean.shape)
        act = mean + std * z
        logp = Normal(mean, std).log_prob(act)
        return jnp.tanh(act), logp, state, mean, std

    def evaluate(
        self,
        key: jax.Array,
        policy_params: Union[hk.Params, Tuple[hk.Params, hk.Params]],
        state: Union[hk.State, Tuple[hk.State, hk.State]],
        obs: jax.Array,
        is_training: bool = True,
    ) -> Tuple[jax.Array, jax.Array]:
        """Sample actions and log-probability terms for algorithm updates."""
        if self.bn:
            velocity_field_params, reference_gn_params = policy_params
            velocity_field_state, reference_gn_state = state
        else:
            velocity_field_params = policy_params
            velocity_field_state = state
            reference_gn_params = None
            reference_gn_state = None

        action, logp, reference_mean, reference_std, action_init = self.flow.sample_with_log_prob(
            key,
            obs,
            velocity_field_params,
            velocity_field_state,
            reference_gn_params,
            reference_gn_state,
            is_training=is_training,
        )
        return action, logp, reference_mean, reference_std, action_init


def create_flow_net(
    key: jax.Array,
    obs_dim: int,
    act_dim: int,
    hidden_sizes: Sequence[int],
    activation: Activation = jax.nn.relu,
    velocity_activation=jax.nn.relu,
    num_timesteps: int = 20,
    template_obs: jax.Array = None,
    template_act: jax.Array = None,
    use_bn: bool = True,
    learn_reference_gn: bool = True,
) -> Tuple[FlowNet, FlowParams]:
    """Create Qflex networks and initialize their parameters."""
    if use_bn:
        q = hk.without_apply_rng(
            hk.transform_with_state(
                lambda obs, act, is_training: QNetBN(hidden_sizes, activation)(obs, act, is_training=is_training)
            )
        )
        velocity_field = hk.without_apply_rng(
            hk.transform_with_state(
                lambda obs, act, t, is_training: FlowVelocityFieldBN(act_dim, hidden_sizes, velocity_activation)(
                    obs, act, t, is_training=is_training
                )
            )
        )
        if learn_reference_gn:
            reference_gn = hk.without_apply_rng(
                hk.transform_with_state(
                    lambda obs, is_training: PolicyNetBN(act_dim, hidden_sizes, activation)(
                        obs, is_training=is_training
                    )
                )
            )
        else:
            reference_gn = None
    else:
        raise NotImplementedError

    @jax.jit
    def init(key, obs, act):
        """Initialize all Qflex parameter/state groups."""
        if use_bn:
            q1_key, q2_key, velocity_field_key, reference_gn_key = jax.random.split(key, 4)
            q1_params, q1_state = q.init(q1_key, obs, act, is_training=True)
            q2_params, q2_state = q.init(q2_key, obs, act, is_training=True)

            target_q1_params = q1_params
            target_q2_params = q2_params

            velocity_field_params, velocity_field_state = velocity_field.init(
                velocity_field_key,
                obs,
                act,
                jnp.zeros((1, 1), dtype=jnp.int32),
                is_training=True,
            )

            reference_gn_params, reference_gn_state = (
                reference_gn.init(reference_gn_key, obs, is_training=True) if reference_gn is not None else (None, None)
            )
            log_alpha = jnp.zeros(act_dim, dtype=jnp.float32)
            exp_prob = 0

            return FlowParams(
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
            )
        else:
            raise NotImplementedError

    sample_obs = jnp.zeros((1, obs_dim), dtype=jnp.float32) if template_obs is None else template_obs
    sample_act = jnp.zeros((1, act_dim), dtype=jnp.float32) if template_act is None else template_act

    params = init(key, sample_obs, sample_act)

    net = FlowNet(
        velocity_field=velocity_field,
        num_timesteps=num_timesteps,
        act_dim=act_dim,
        q=q.apply,
        target_entropy=-1,
        bn=use_bn,
        reference_gn=reference_gn if reference_gn is not None else None,
    )

    return net, params
