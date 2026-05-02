from typing import Callable, Optional
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import jax.random as random
import haiku as hk
from jax.flatten_util import ravel_pytree
from jax.scipy.stats.norm import logpdf as norm_logpdf

PRNGKey = jax.random.PRNGKey


def simple_euler(func, x0, t, obs, bound_dim: int = 0) -> list:
    """Integrate a vector field with a clipped Euler solver.

    Args:
        func: Callable with signature ``func(obs, x, t)``.
        x0: Initial action or augmented state.
        t: 1-D time grid.
        obs: Observation passed to the vector field.
        bound_dim: Kept for compatibility with existing solver calls.

    Returns:
        The scanned trajectory over ``t[1:]``.
    """
    dt = t[1] - t[0]
    is_tuple = isinstance(x0, tuple)
    x = x0
    flow = [x]

    def body_fn(x, ti):
        dx = func(obs, x, ti)
        if is_tuple:
            x, vs = x
        # Clip each Euler step to avoid unstable jumps during sampling.
        dx = dx.clip(-1 / dt, 1 / dt)
        x = x + dt * dx

        if is_tuple:
            x = (x, vs)
        return x, x

    x, flow = jax.lax.scan(body_fn, x, t[1:])
    return flow


def _sample_probe(key, shape, dist="rademacher"):
    """Sample Hutchinson trace-estimator probe vectors."""
    if dist == "rademacher":
        return jax.random.choice(key, jnp.array([-1.0, 1.0]), shape=shape)
    if dist == "gaussian":
        return jax.random.normal(key, shape)
    raise ValueError("dist must be 'rademacher' or 'gaussian'")


@dataclass(frozen=True)
class FlowMatching:
    """Flow matching sampler and likelihood helper.

    Attributes:
        velocity_field_apply: Parameterized vector-field apply function.
        act_dim: Action-space dimension.
        num_timesteps: Number of ODE steps used for sampling.
        sigma: Standard deviation for reference Gaussian noise.
        odeint: ODE solver used by the sampler.
        reference_gn: Optional learned Gaussian reference policy.
        use_bn: Whether the vector field has Haiku state, such as batch norm.
    """

    velocity_field_apply: Callable
    act_dim: int
    num_timesteps: int
    sigma: float = 1e-4
    odeint: Callable = simple_euler
    reference_gn: Optional[Callable] = None
    use_bn: bool = True

    def sample_fn(
        self,
        vector_field_param: hk.Params,
        reference_gn_param: Optional[hk.Params] = None,
    ) -> Callable:
        """Build a single-observation sampler for stateless vector fields."""
        def vector_field(obs, act, time):
            return self.velocity_field_apply(vector_field_param, obs, act, time)

        if self.reference_gn is not None:
            assert (
                reference_gn_param is not None
            ), "Reference Gaussian network parameters must be provided when using reference_gn."

            def reference_dist(obs):
                return self.reference_gn.apply(reference_gn_param, obs)

        def one_sample(key: PRNGKey, obs: jax.Array):
            x0 = random.normal(key, (self.act_dim,))
            if self.reference_gn is not None:
                mean, std = reference_dist(obs)
                x0 = mean + std * x0
            x0 = jnp.tanh(x0)

            flow = self.odeint(vector_field, x0, jnp.linspace(0, 1, self.num_timesteps + 1), obs, x0.shape[0])
            return flow, x0

        return one_sample

    def sample_bn_fn(
        self,
        vector_field_param: hk.Params,
        vector_field_state: hk.State,
        reference_gn_param: Optional[hk.Params] = None,
        reference_gn_state: Optional[hk.State] = None,
        is_training: bool = False,
    ) -> Callable:
        """Build a single-observation sampler for stateful vector fields."""
        def vector_field(obs, act, time):
            return self.velocity_field_apply(
                vector_field_param,
                vector_field_state,
                obs,
                act,
                jnp.array([time], dtype=jnp.float32),
                is_training,
            )[0][0]

        if self.reference_gn is not None and reference_gn_param is not None:

            def reference_dist(obs):
                return self.reference_gn.apply(reference_gn_param, reference_gn_state, obs, is_training)

        def one_sample(key: PRNGKey, obs: jax.Array):
            x0 = random.normal(key, (self.act_dim,))
            if reference_gn_param is not None:
                (mean, std), _ = reference_dist(obs)
                x0 = mean + std * x0
                x0 = x0.flatten()
            x0 = jnp.tanh(x0)

            flow = self.odeint(vector_field, x0, jnp.linspace(0, 1, self.num_timesteps + 1), obs, x0.shape[0])
            return flow[-1], x0

        return one_sample

    def sample(
        self,
        rng_key: PRNGKey,
        obs: jax.Array,
        vector_field_param: hk.Params,
        vector_field_state: Optional[hk.State] = None,
        reference_gn_param: Optional[hk.Params] = None,
        reference_gn_state: Optional[hk.State] = None,
        is_training: Optional[bool] = None,
    ) -> jax.Array:
        """Sample a batch of actions by integrating the learned flow."""
        if self.use_bn:
            assert is_training is not None, "is_training must be provided when using batch normalization."
            one_sample = self.sample_bn_fn(
                vector_field_param, vector_field_state, reference_gn_param, reference_gn_state, is_training
            )
            if obs.ndim == 1:
                obs = obs.reshape(1, -1)
            keys = random.split(rng_key, obs.shape[0])
            return jax.vmap(one_sample)(keys, obs)
        else:
            one_sample = self.sample_fn(vector_field_param, reference_gn_param)
            if obs.ndim == 1:
                obs = obs.reshape(1, -1)
            keys = random.split(rng_key, obs.shape[0])

            return jax.vmap(one_sample)(keys, obs)

    def sample_with_log_prob_fn(
        self,
        vector_field_param: jax.Array,
        reference_gn_param: Optional[hk.Params] = None,
    ) -> Callable:
        """Build a stateless sampler that also estimates log probability."""
        if self.reference_gn is not None:
            assert (
                reference_gn_param is not None
            ), "Reference Gaussian network parameters must be provided when using reference_gn."

            def reference_dist(obs):
                return self.reference_gn.apply(reference_gn_param, obs)

        def augmented_vector_field(combined_input, time):
            obs, x_ldj = combined_input
            x = x_ldj[:-1]
            dx = self.velocity_field_apply(vector_field_param, obs, x, time)
            jacobian = jax.jacfwd(lambda obs, act, time: self.velocity_field_apply(vector_field_param, obs, act, time))(
                x
            )
            dldj = -jnp.einsum("ii", jacobian)
            return jnp.concatenate([dx, jnp.array([dldj])])

        def logprob_fn(key: PRNGKey, obs: jax.Array):
            sample = jax.random.normal(key, (self.act_dim,))
            p0 = norm_logpdf(sample).sum()
            if self.reference_gn is not None:
                mean, std = reference_dist(obs)
                sample = mean + std * sample
                mean = mean.flatten()
                std = std.flatten()
            else:
                mean = jnp.ones_like(sample)
                std = jnp.zeros_like(sample)
            sample = jnp.tanh(sample)
            flow = self.odeint(
                augmented_vector_field,
                (obs, jnp.concatenate([sample, jnp.array([p0])])),
                jnp.linspace(0, 1, self.num_timesteps + 1),
                obs,
                sample.shape[0],
            )
            inv_flow = flow[-1, :-1]
            ldj_flow = flow[-1, -1]
            return inv_flow, ldj_flow, mean, std

        return logprob_fn

    def sample_bn_with_log_prob_fn(
        self,
        vector_field_param: jax.Array,
        vector_field_state: hk.State,
        reference_gn_param: Optional[hk.Params] = None,
        reference_gn_state: Optional[hk.State] = None,
        is_training: bool = False,
    ) -> Callable:
        """Build a stateful sampler that also estimates log probability."""
        if self.reference_gn is not None:
            assert (
                reference_gn_param is not None
            ), "Reference Gaussian network parameters must be provided when using reference_gn."

            def reference_dist(obs):
                return self.reference_gn.apply(reference_gn_param, reference_gn_state, obs, is_training)

        def augmented_vector_field(obs, combined_input, time):
            x_ldj, vs = combined_input
            x = x_ldj[:-1]
            time = jnp.array([time], dtype=jnp.float32)

            dx, _ = self.velocity_field_apply(vector_field_param, vector_field_state, obs, x, time, is_training)
            dx = dx[0]

            def get_velocity(x):
                dx, _ = self.velocity_field_apply(vector_field_param, vector_field_state, obs, x, time, is_training)
                return dx

            def one_sample(v):
                # Hutchinson estimator for the trace of the velocity Jacobian.
                _, jvp = jax.jvp(get_velocity, (x,), (v,))
                return jnp.dot(v, jvp.flatten())

            per_sample = jax.vmap(jax.jit(one_sample))(vs)
            dldj = -per_sample.mean()
            return jnp.concatenate([dx, jnp.array([dldj])])

        def logprob_fn(key: PRNGKey, obs: jax.Array):
            sample = jax.random.normal(key, (self.act_dim,))
            p0 = norm_logpdf(sample).sum()
            if self.reference_gn is not None:
                (mean, std), _ = reference_dist(obs)
                sample = mean + std * sample
                mean = mean.flatten()
                std = std.flatten()
            else:
                mean = jnp.ones_like(sample)
                std = jnp.zeros_like(sample)
            sample = jnp.tanh(sample)
            sample, _ = ravel_pytree(sample)
            tanh_trans_prob = jnp.clip(1 - sample.clip(-1 + 1e-6, 1 - 1e-6), -5, 5).sum()
            p0 -= tanh_trans_prob
            keys = jax.random.split(key, 16)
            vs = jax.vmap(lambda k: _sample_probe(k, sample.shape))(keys)

            flow = self.odeint(
                augmented_vector_field,
                (jnp.concatenate([sample, jnp.array([p0])]), vs),
                jnp.linspace(0, 1, self.num_timesteps + 1),
                obs,
                sample.shape[0],
            )

            flow = flow[0][-1]
            inv_flow = flow[:-1]
            ldj_flow = flow[-1]
            return inv_flow, ldj_flow, mean, std, sample

        return logprob_fn

    def sample_with_log_prob(
        self,
        rng_key: PRNGKey,
        obs: jax.Array,
        vector_field_param: hk.Params,
        vector_field_state: Optional[hk.State] = None,
        reference_gn_param: Optional[hk.Params] = None,
        reference_gn_state: Optional[hk.State] = None,
        is_training: Optional[bool] = None,
    ) -> jax.Array:
        """Sample a batch of actions and associated log-probability terms."""
        if self.use_bn:
            assert is_training is not None, "is_training must be provided when using batch normalization."
            one_sample = self.sample_bn_with_log_prob_fn(
                vector_field_param,
                vector_field_state,
                reference_gn_param,
                reference_gn_state,
                is_training,
            )
            keys = random.split(rng_key, obs.shape[0])
            return jax.vmap(one_sample)(keys, obs)
        else:
            one_sample = self.sample_with_log_prob_fn(vector_field_param, reference_gn_param)
            keys = random.split(rng_key, obs.shape[0])
            return jax.vmap(one_sample)(keys, obs)
