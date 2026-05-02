from dataclasses import dataclass
from typing import Callable, NamedTuple, Sequence, Tuple
from numpyro.distributions import Normal
import jax, jax.numpy as jnp
import haiku as hk
from relax.network.common import WithSquashedGaussianPolicy
from relax.network.blocks import Activation, QNetBN, PolicyNetBN
from relax.env import RelaxWrapper

# from relax.network.sac import SACNet, SACParams

class CrossQParams(NamedTuple):
    q1: hk.Params
    q2: hk.Params
    target_q1: hk.Params
    target_q2: hk.Params
    policy: hk.Params
    q1_state: hk.State
    q2_state: hk.State
    policy_state: hk.State
    log_alpha: jax.Array
    # Eval
    # q1_eval : hk.Params
    # q2_eval : hk.Params
    # policy_eval : hk.Params
    # q1_eval_state: hk.State
    # q2_eval_state: hk.State
    # policy_eval_state: hk.State
    

@dataclass
class CrossQNet:
    policy: Callable[[hk.Params, jax.Array], Tuple[jax.Array, jax.Array]]
    # policy_eval: Callable[[hk.Params, jax.Array], Tuple[jax.Array, jax.Array]]
    q: Callable[[hk.Params, jax.Array, jax.Array], jax.Array]
    target_entropy: float
    bn: bool = True # Need to use state when using BN in CrossQ

    def get_action(self, key: jax.Array, policy_params: hk.Params, state: hk.State, obs: jax.Array) -> jax.Array:
        """for data collection"""
        (mean, std), state = self.policy(policy_params, state, obs, is_training=False)
        z = jax.random.normal(key, mean.shape)
        act = mean + std * z
        return jnp.tanh(act)

    

    def get_deterministic_action(self, policy_params: hk.Params, state: hk.State, obs: jax.Array) -> jax.Array:
        """for evaluation"""
        (mean, _), state = self.policy(policy_params, state, obs, is_training=False)
        return jnp.tanh(mean)

    # def get_action_eval(self, key: jax.Array, policy_params: hk.Params, state: hk.State, obs: jax.Array) -> jax.Array:
    #     """for data collection"""
    #     (mean, std), state = self.policy(policy_params, state, obs, is_training=False)
    #     z = jax.random.normal(key, mean.shape)
    #     act = mean + std * z
    #     return jnp.tanh(act)

    

    # def get_deterministic_action_eval(self, policy_params: hk.Params, state: hk.State, obs: jax.Array) -> jax.Array:
    #     """for evaluation"""
    #     (mean, _), state = self.policy(policy_params, state, obs, is_training=False)
    #     return jnp.tanh(mean)
    


    def evaluate(
        self, key: jax.Array, policy_params: hk.Params, state: hk.State, obs: jax.Array, is_training: bool = True
    ) -> Tuple[jax.Array, jax.Array]:
        """for algorithm update"""
        (mean, std), state = self.policy(policy_params, state, obs, is_training=is_training)
        z = jax.random.normal(key, mean.shape)
        act = mean + std * z
        logp = Normal(mean, std).log_prob(act) # - 2 * (math.log(2) - act - jax.nn.softplus(-2 * act))
        return jnp.tanh(act), logp.sum(axis=-1), state


def create_crossq_net(
    key: jax.Array,
    obs_dim: int,
    act_dim: int,
    hidden_sizes: Sequence[int],
    activation: Activation = jax.nn.relu,
    template_obs: jax.Array = None,
    template_act: jax.Array = None,
) -> Tuple[CrossQNet, CrossQParams]:
    # Use batch normalization in CrossQ
    # q = hk.without_apply_rng( hk.transform(lambda obs, act: QNetBN(hidden_sizes, activation)(obs, act)))
    # policy = hk.without_apply_rng(hk.transform(lambda obs: PolicyNetBN(act_dim, hidden_sizes, activation)(obs)))
    # q = hk.without_apply_rng(hk.transform_with_state(lambda obs, act: QNetBN(hidden_sizes, activation)(obs, act, is_training=True)))
    # policy = hk.without_apply_rng(hk.transform_with_state(lambda obs: PolicyNetBN(act_dim, hidden_sizes, activation)(obs, is_training=True)))
    q = hk.without_apply_rng(hk.transform_with_state(lambda obs, act, is_training: QNetBN(hidden_sizes, activation)(obs, act, is_training=is_training)))
    policy = hk.without_apply_rng(hk.transform_with_state(lambda obs, is_training: PolicyNetBN(act_dim, hidden_sizes, activation)(obs, is_training=is_training)))


    # q = hk.without_apply_rng(hk.transform_with_state(QNetBN(hidden_sizes, activation)))
    # policy = hk.without_apply_rng(hk.transform_with_state(PolicyNetBN(act_dim, hidden_sizes, activation)))


    # q_eval = hk.without_apply_rng(hk.transform_with_state(lambda obs, act: QNetBN(hidden_sizes, activation)(obs, act, is_training=False)))
    # policy_eval = hk.without_apply_rng(hk.transform_with_state(lambda obs: PolicyNetBN(act_dim, hidden_sizes, activation)(obs, is_training=False)))

    @jax.jit
    def init(key, obs, act):
        q1_key, q2_key, policy_key = jax.random.split(key, 3)
        # q.set_state("is_training", True)
        q1_params, q1_state = q.init(q1_key, obs, act, is_training=True)
        q2_params, q2_state = q.init(q2_key, obs, act, is_training=True)
        # q1_eval_params, q1_eval_state = q_eval.init(q1_key, obs, act)
        # q2_eval_params, q2_eval_state = q_eval.init(q2_key, obs, act)

        target_q1_params = q1_params
        target_q2_params = q2_params
        policy_params, policy_state = policy.init(policy_key, obs, is_training=True)
        # policy_eval_params, policy_eval_state = policy_eval.init(policy_key, obs)
        log_alpha = jnp.array(0.0, dtype=jnp.float32)
        return CrossQParams(q1_params, q2_params, target_q1_params, target_q2_params, policy_params, 
                         q1_state, q2_state, policy_state, log_alpha,
                        #  q1_eval_params, q2_eval_params, policy_eval_params, q1_eval_state, q2_eval_state, policy_eval_state
                         )

    
    sample_obs = jnp.zeros((1, obs_dim), dtype=jnp.float32) if template_obs is None else template_obs
    sample_act = jnp.zeros((1, act_dim), dtype=jnp.float32) if template_act is None else template_act
    print('temp shape', sample_obs.shape, sample_act.shape)
    # jax.debug.print(f'{sample_obs[:5]}, {sample_act[:5]}')
    # raise NotImplementedError
    # sample_is_training = jnp.ones(1, dtype=jnp.bool)
    params = init(key, sample_obs, sample_act)
    net = CrossQNet(policy=policy.apply, 
                    # policy_eval=policy_eval.apply, 
                    q=q.apply, target_entropy=-act_dim)

    return net, params