import os
from pathlib import Path
import pickle



from typing import NamedTuple, Tuple
import numpy as np
import jax, jax.numpy as jnp
import optax
import haiku as hk

from relax.algorithm.base import Algorithm
# from relax.network.sac import SACNet, SACParams
from relax.utils.experience import Experience
from relax.utils.typing_utils import Metric
# from relax.algorithm.sac import SAC, SACOptStates, SACTrainState
# from relax.network.crossq import CrossQNet, CrossQParams
from relax.network.crossq_langevin import CrossQLangevinNet, CrossQLangevinParams
from relax.utils.experience import Experience
from relax.utils.persistence import make_persist
from relax.utils.typing_utils import Metric

class CrossQOptStates(NamedTuple):
    q1: optax.OptState
    q2: optax.OptState
    policy: optax.OptState
    # q1_state: optax.OptState
    # q2_state: optax.OptState
    # policy_state: optax.OptState
    log_alpha: optax.OptState


class CrossQTrainState(NamedTuple):
    params: CrossQLangevinParams
    opt_state: CrossQOptStates


class CrossQLangevin(Algorithm):
    def __init__(self, agent: CrossQLangevinNet, params: CrossQLangevinParams, *, gamma: float = 0.99, lr: float = 1e-4,
                 alpha_lr: float = 3e-4, 
                 tau: float = 1.0, # Not use target Q function
                 reward_scale: float = 1,):
        self.agent = agent
        self.gamma = gamma
        self.tau = tau
        self.policy_optim = optax.adam(lr, 
                                       b1=0.5
                                       )
        self.q_optim = optax.adam(lr, 
                                  b1=0.5
                                  )
        self.log_alpha_optim = optax.adam(alpha_lr)
        self.reward_scale = reward_scale
        self.updated_policy_structure = False
        self.state = CrossQTrainState(
            params=params,
            opt_state=CrossQOptStates(
                q1=self.q_optim.init(params.q1),
                q2=self.q_optim.init(params.q2),
                policy=self.policy_optim.init(params.policy),
                log_alpha=self.log_alpha_optim.init(params.log_alpha),
            ),
        )
    
        @jax.jit
        def stateless_update(
            key: jax.Array, state: CrossQTrainState, data: Experience
        ) -> Tuple[CrossQTrainState, Metric]:
            obs, action, reward, next_obs, done = data.obs, data.action, data.reward, data.next_obs, data.done
            obs = jnp.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
            next_obs = jnp.nan_to_num(next_obs, nan=0.0, posinf=0.0, neginf=0.0)
            action = jnp.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
            reward = jnp.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
            q1_params, q2_params, target_q1_params, target_q2_params, policy_params, q1_state, q2_state, policy_state, log_alpha = state.params
                # q1_eval_params, q2_eval_params, policy_eval_params, q1_eval_state, q2_eval_state, policy_eval_state 
            
            q1_opt_state, q2_opt_state, policy_opt_state, log_alpha_opt_state = state.opt_state
            next_eval_key, new_eval_key = jax.random.split(key)

            reward *= self.reward_scale

            # compute target q
            next_action, next_logp, _ = self.agent.evaluate(next_eval_key, policy_params, policy_state, next_obs, is_training=False)
            # q1_target = self.agent.q(target_q1_params, next_obs, next_action)
            # q2_target = self.agent.q(target_q2_params, next_obs, next_action)
            # q_target = jnp.minimum(q1_target, q2_target) - jnp.exp(log_alpha) * next_logp
            # q_backup = reward + (1 - done) * self.gamma * q_target
            # print(q1_state['q_net_bn/batch_norm/~/mean_ema'].keys())
            # print('q1', q1_state['q_net_bn/batch_norm/~/mean_ema']['average'][0, :5], 
            #       q1_state['q_net_bn/batch_norm/~/var_ema']['average'].max(), 
            #       q1_state['q_net_bn/batch_norm/~/var_ema']['average'].min(), 
            #       q1_state['q_net_bn/batch_norm/~/mean_ema']['counter'])
            # Concat current and next in CrossQ to avoid distribution shift
            catted_q1, _ = self.agent.q(
                q1_params,
                q1_state,
                jnp.concatenate([obs, next_obs], axis=0), 
                jnp.concatenate([action, next_action], axis=0),
                is_training=True
            )
            # print(q1_state)
            # raise NotImplementError
            # print('q1 after', q1_state['q_net_bn/batch_norm/~/mean_ema']['average'][0, :5], 
            #       q1_state['q_net_bn/batch_norm/~/var_ema']['average'].max(), 
            #       q1_state['q_net_bn/batch_norm/~/var_ema']['average'].min(), 
            #       q1_state['q_net_bn/batch_norm/~/mean_ema']['counter'])

            catted_q2, _ = self.agent.q(
                q2_params,
                q2_state,
                jnp.concatenate([obs, next_obs], axis=0), 
                jnp.concatenate([action, next_action], axis=0),
                is_training=True
            )
            
            _, q1_target = jnp.split(catted_q1, 2)
            _, q2_target = jnp.split(catted_q2, 2)
            
            q_target = jnp.minimum(q1_target, q2_target) - jnp.exp(log_alpha) * next_logp
            q_backup = reward + (1 - done) * self.gamma * q_target


            # update q
            def q_loss_fn(q_params: hk.Params, q_state: hk.State) -> jax.Array:
                catted_q, q_state = self.agent.q(
                    q_params,
                    q_state,
                    jnp.concatenate([obs, next_obs], axis=0), 
                    jnp.concatenate([action, next_action], axis=0),
                    is_training=True
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

            # update policy
            def policy_loss_fn(policy_params: hk.Params, policy_state: hk.State, q1_state:hk.State, q2_state:hk.State) -> jax.Array:
                new_action, new_logp, policy_state = self.agent.evaluate(new_eval_key, policy_params, policy_state, obs, is_training=True)
                q1, _  = self.agent.q(target_q1_params, q1_state, obs, new_action, is_training=False)
                q2, _ = self.agent.q(target_q2_params, q2_state, obs, new_action, is_training=False)
                
                q = jnp.minimum(q1, q2)
                policy_loss = jnp.mean(jnp.exp(log_alpha) * new_logp - q)
                return policy_loss, (q1, q2, policy_state, new_logp)

            (policy_loss, aux), policy_grads = jax.value_and_grad(policy_loss_fn, has_aux=True)(policy_params, policy_state, q1_state, q2_state)
            q1, q2, policy_state, new_logp = aux
            policy_update, policy_opt_state = self.policy_optim.update(policy_grads, policy_opt_state)
            policy_params = optax.apply_updates(policy_params, policy_update)
            # jax.debug.print(f'{jax.tree.structure(policy_params)}')
            # update alpha
            def log_alpha_loss_fn(log_alpha: jax.Array) -> jax.Array:
                log_alpha_loss = -jnp.mean(log_alpha * (new_logp + self.agent.target_entropy))
                return log_alpha_loss

            log_alpha_grads = jax.grad(log_alpha_loss_fn)(log_alpha)
            log_alpha_update, log_alpha_opt_state = self.log_alpha_optim.update(log_alpha_grads, log_alpha_opt_state)
            log_alpha = optax.apply_updates(log_alpha, log_alpha_update)

            # update target q
            target_q1_params = optax.incremental_update(q1_params, target_q1_params, self.tau)
            target_q2_params = optax.incremental_update(q2_params, target_q2_params, self.tau)

            # update eval params
            # q1_eval_params = optax.incremental_update(q1_params, q1_eval_params, 1)
            # q2_eval_params = optax.incremental_update(q2_params, q2_eval_params, 1)
            # policy_eval_params = optax.incremental_update(policy_params, policy_eval_params, 1)

            state = CrossQTrainState(
                params=CrossQLangevinParams(q1_params, q2_params, target_q1_params, target_q2_params, policy_params, q1_state, q2_state, policy_state, log_alpha,
                                    # q1_eval_params, q2_eval_params, policy_eval_params, q1_eval_state, q2_eval_state, policy_eval_state
                                    ),
                opt_state=CrossQOptStates(q1_opt_state, q2_opt_state, policy_opt_state, log_alpha_opt_state),
            )
            info = {
                "q1_loss": q1_loss,
                "q2_loss": q2_loss,
                "q1": jnp.mean(q1),
                "q2": jnp.mean(q2),
                "policy_loss": policy_loss,
                "entropy": -jnp.mean(new_logp),
                "alpha": jnp.exp(log_alpha),
                # "q1_bn_mean1_max": q1_state['q_net_bn/batch_renorm/~/mean_ema']['average'].max(),
                # "q1_bn_mean1_min": q1_state['q_net_bn/batch_renorm/~/mean_ema']['average'].min(),
                # "q1_bn_var1_max": q1_state['q_net_bn/batch_renorm/~/var_ema']['average'].max(),
                # "q1_bn_var1_min": q1_state['q_net_bn/batch_renorm/~/var_ema']['average'].min(),
                # "q1_bn_mean2_max": q1_state['q_net_bn/batch_renorm_1/~/mean_ema']['average'].max(),
                # "q1_bn_mean2_min": q1_state['q_net_bn/batch_renorm_1/~/mean_ema']['average'].min(),
                # "q1_bn_var2_max": q1_state['q_net_bn/batch_renorm_1/~/var_ema']['average'].max(),
                # "q1_bn_var2_min": q1_state['q_net_bn/batch_renorm_1/~/var_ema']['average'].min(),
                # "q1_bn_mean3_max": q1_state['q_net_bn/batch_renorm_2/~/mean_ema']['average'].max(),
                # "q1_bn_mean3_min": q1_state['q_net_bn/batch_renorm_2/~/mean_ema']['average'].min(),
                # "q1_bn_var3_max": q1_state['q_net_bn/batch_renorm_2/~/var_ema']['average'].max(),
                # "q1_bn_var3_min": q1_state['q_net_bn/batch_renorm_2/~/var_ema']['average'].min(),
                # "q1_bn_mean4_max": q1_state['q_net_bn/batch_renorm_3/~/mean_ema']['average'].max(),
                # "q1_bn_mean4_min": q1_state['q_net_bn/batch_renorm_3/~/mean_ema']['average'].min(),
                # "q1_bn_var4_max": q1_state['q_net_bn/batch_renorm_3/~/var_ema']['average'].max(),
                # "q1_bn_var4_min": q1_state['q_net_bn/batch_renorm_3/~/var_ema']['average'].min(),
                #  "q2_bn_mean1_max": q2_state['q_net_bn/batch_renorm/~/mean_ema']['average'].max(),
                # "q2_bn_mean1_min": q2_state['q_net_bn/batch_renorm/~/mean_ema']['average'].min(),
                # "q2_bn_var1_max": q2_state['q_net_bn/batch_renorm/~/var_ema']['average'].max(),
                # "q2_bn_var1_min": q2_state['q_net_bn/batch_renorm/~/var_ema']['average'].min(),
                # "q2_bn_mean2_max": q2_state['q_net_bn/batch_renorm_1/~/mean_ema']['average'].max(),
                # "q2_bn_mean2_min": q2_state['q_net_bn/batch_renorm_1/~/mean_ema']['average'].min(),
                # "q2_bn_var2_max": q2_state['q_net_bn/batch_renorm_1/~/var_ema']['average'].max(),
                # "q2_bn_var2_min": q2_state['q_net_bn/batch_renorm_1/~/var_ema']['average'].min(),
                # "q2_bn_mean3_max": q2_state['q_net_bn/batch_renorm_2/~/mean_ema']['average'].max(),
                # "q2_bn_mean3_min": q2_state['q_net_bn/batch_renorm_2/~/mean_ema']['average'].min(),
                # "q2_bn_var3_max": q2_state['q_net_bn/batch_renorm_2/~/var_ema']['average'].max(),
                # "q2_bn_var3_min": q2_state['q_net_bn/batch_renorm_2/~/var_ema']['average'].min(),
                # "q2_bn_mean4_max": q2_state['q_net_bn/batch_renorm_3/~/mean_ema']['average'].max(),
                # "q2_bn_mean4_min": q2_state['q_net_bn/batch_renorm_3/~/mean_ema']['average'].min(),
                # "q2_bn_var4_max": q2_state['q_net_bn/batch_renorm_3/~/var_ema']['average'].max(),
                # "q2_bn_var4_min": q2_state['q_net_bn/batch_renorm_3/~/var_ema']['average'].min(),
            }

            # if not self.updated_policy_structure:
            #     self.save_policy_structure(self.root_path, dummy_obs=obs[0])
            #     self.updated_policy_structure = True
            #     print("Policy structure saved to 'policy_structure' directory.")

            return state, info

        self._implement_common_behavior(stateless_update, self.agent.get_action, self.agent.get_deterministic_action)
        # self._get_action_eval = jax.jit(self.agent.get_action_eval)
        # self._get_deterministic_action_eval = jax.jit(self.agent.get_deterministic_action_eval)

        
        
    
    def warmup(self, data: Experience) -> None:
        key = jax.random.key(0)
        obs = data.obs[0]
        self.input_shape = obs.shape
        policy_params = self.get_policy_params()
        # policy_state = self.state.params.policy_state
        policy_state = self.get_policy_state()
        self._update(key, self.state, data)
        # print('update')
        # raise NotImplementedError
        # policy_params = self.get_policy_params()
        # # policy_state = self.state.params.policy_state
        # policy_state = self.get_policy_state()
        self._get_action(key, policy_params, policy_state, obs)
        self._get_deterministic_action(policy_params, policy_state, obs)

    def get_action(self, key: jax.Array, obs: np.ndarray) -> np.ndarray:
        # policy_state = self.state.params.policy_state
        action = self._get_action(key, self.get_policy_params(), self.get_policy_state(), obs)
        return np.asarray(action)

    def get_deterministic_action(self, obs: np.ndarray) -> np.ndarray:
        # policy_state = self.state.params.policy_state
        action = self._get_deterministic_action(self.get_policy_params(), self.get_policy_state(), obs)
        return np.asarray(action)

    def save_policy_structure(self, root: os.PathLike, dummy_obs: jax.Array) -> None:
        self.root_path = root
        root = Path(root)
        policy_state = self.get_policy_state()
        key = jax.random.key(0)
        stochastic = make_persist(self._get_action._fun)(key, self.get_policy_params(), policy_state, dummy_obs)
        deterministic = make_persist(self._get_deterministic_action._fun)(self.get_policy_params(), policy_state, dummy_obs)

        stochastic.save(root / "stochastic.pkl")
        stochastic.save_info(root / "stochastic.txt")
        deterministic.save(root / "deterministic.pkl")
        deterministic.save_info(root / "deterministic.txt")
    
    def get_policy_state(self):
        return (self.state.params.policy_state, self.state.params.q1_state, self.state.params.q2_state)

    def get_policy_params(self):
        return (self.state.params.policy, self.state.params.q1, self.state.params.q2)

    def save_policy(self, path: str) -> None:
        # update policy structure
        
        policy = jax.device_get(self.get_policy_params())
        policy_state = jax.device_get(self.get_policy_state())
        with open(path, "wb") as f:
            pickle.dump((policy, policy_state), f)
        
