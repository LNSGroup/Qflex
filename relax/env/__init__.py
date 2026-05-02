import sys
import os
import numpy as np
from gymnasium import Env, Wrapper, make

from gymnasium.spaces import Box
try:
    import myosuite
    from myosuite.utils import gym
except:
    pass # Not use myosuite, just ignore the import error


try:
    import humanoid_bench
except:
    pass # Not use humanoid_bench, just ignore the import error

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import envs

from relax.env.vector import VectorEnv, SerialVectorEnv, GymProcessVectorEnv, PipeProcessVectorEnv, SpinlockProcessVectorEnv, FutexProcessVectorEnv

class RelaxWrapper(Wrapper):
    render_mode = 'rgb_array'
    def __init__(self, env: Env, action_seed: int = 0):
        super().__init__(env)
        self.env: Env[np.ndarray, np.ndarray]
        # if self.render_mode is None:
        #     # self.render_mode = "rgb_array"
        #     setattr(self, "render_mode", "rgb_array")
        
        assert isinstance(env.observation_space, Box)
        assert isinstance(env.action_space, Box) and env.action_space.is_bounded()
        if isinstance(env, VectorEnv):
            _, self.obs_dim = env.observation_space.shape
            _, self.act_dim = env.action_space.shape
            single_action_space = env.single_action_space
        else:
            self.obs_dim, = env.observation_space.shape
            self.act_dim, = env.action_space.shape
            single_action_space = env.action_space

        if np.any(single_action_space.low != -1.0) or np.any(single_action_space.high != 1.0):
            print(f"NOTE: The action space is not normalized, but {single_action_space.low} to {single_action_space.high}, will be rescaled.")
            self.needs_rescale = True
            self.original_action_center = (single_action_space.low + single_action_space.high) * 0.5
            self.original_action_half_range = (single_action_space.high - single_action_space.low) * 0.5
        else:
            self.needs_rescale = False
        self.original_action_dtype = env.action_space.dtype

        self._action_space = Box(
            low=-1,
            high=1,
            shape=env.action_space.shape,
            dtype=np.float32,
            seed=action_seed
        )

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return obs.astype(np.float32, copy=False), info

    def step(self, action: np.ndarray):
        action = action.astype(self.original_action_dtype)
        action = action.clip(-1, 1)
        if self.needs_rescale:
            action *= self.original_action_half_range
            action += self.original_action_center
        obs, reward, terminated, truncated, info = self.env.step(action)
        # find nan in reward
        # if np.isnan(reward).any():
        #     print(f"Warning: NaN in reward: {reward}")
        #     reward = np.nan_to_num(reward, nan=0.0)
        #     truncated = True
        #     terminated = True
        # if np.isnan(obs).any():
        #     print(f"Warning: NaN in observation: {obs}")
        #     obs = np.nan_to_num(obs, nan=0.0)
        #     truncated = True
        #     terminated = True
        
        return obs.astype(np.float32, copy=False), reward, terminated, truncated, info

    def render(self):
        try:
            return super().render()
        except:
            return self.env.unwrapped.mj_renderer.render_offscreen(width=400,height=400,camera_id=-1)

def create_env(name: str, seed: int, action_seed: int = 0):
    try:
        env = make(name, render_mode="rgb_array")
    except Exception as e:
        print('env err', e)
        env = gym.make(name)
    try:
        env.reset(seed=seed)
        env = RelaxWrapper(env, action_seed)
    except:
        print('env2 err')
        raise NotImplementedError
    return env, env.obs_dim, env.act_dim

def create_vector_env(name: str, num_envs: int, seed: int, action_seed: int = 0, mode: str = "serial", **kwargs):
    Impl = {
        "serial": SerialVectorEnv,
        "gym": GymProcessVectorEnv,
        "pipe": PipeProcessVectorEnv,
        "spinlock": SpinlockProcessVectorEnv,
        "futex": FutexProcessVectorEnv,
    }[mode]
    env = Impl(name, num_envs, seed, **kwargs)
    env = RelaxWrapper(env, action_seed)
    return env, env.obs_dim, env.act_dim
