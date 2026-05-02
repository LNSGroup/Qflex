import os
import collections
from typing import Dict, Any, Optional

import numpy as np
import mujoco

from envs.import_gym import *
from envs.utils import get_observation_space, get_render_fps, action_obs_check, quat2mat

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 13.0,
    "lookat": np.array((2, 0.0, 0)),
    "elevation": -20.0,
    "azimuth": 120
}


class OstrichRunEnvV1(MujocoEnv, EzPickle):
    metadata: Dict[str, Any] = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 10,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        forward_reward_weight=10,
        reset_noise_scale=0,
        model_path: str = "",
        skip_frames: int = 10,
        **kwargs
    ):
    
        # model_path must be abspath
        model_path = os.path.dirname(__file__) + "/ostrichrl/ostrichrl/assets/models/ostrich/ostrich.xml"
        model_path = os.path.abspath(model_path)


        assert render_mode is None or render_mode in self.metadata["render_modes"]

        observation_space = get_observation_space(
            model_path,
            self._get_obs_core,
            {
            },
            # use_actuator_filter=False
        )

        fps = get_render_fps(model_path, skip_frames)
        self.metadata["render_fps"] = fps
        self.control_timestep = 1 / fps

        EzPickle.__init__(
            self,
            render_mode,
            forward_reward_weight,
            reset_noise_scale,
            **kwargs
        )

        self.render_mode = render_mode
        self._reset_noise_scale = reset_noise_scale

    
        MujocoEnv.__init__(
            self, model_path, skip_frames, observation_space=observation_space, render_mode=render_mode, camera_name="pelvis_camera", **kwargs
        )
        action_obs_check(self)

        # print("observation space shape: ", self.observation_space.shape)
        # print("action space shape: ", self.action_space.shape)
        
        
        # self.init_qpos[:] = self.model.key_qpos[0].copy()
        # self.init_qvel[:] = self.model.key_qvel[2].copy()
        self.body_name_list = [self.model.body(body_id).name for body_id in range(self.model.nbody)]
        self.geom_name_list = [self.model.geom(geom_id).name for geom_id in range(self.model.ngeom)]
        self.sensor_name_list = [self.model.sensor(ss_id).name for ss_id in range(self.model.nsensor)]

    def seed(self, seed=0):
        pass

    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    @property
    def terminated(self):
        terminated = self._get_done()
        return terminated

    def get_obs(self):

        head_height = np.array([self.head_height()])
        pelvis_height = np.array([self.pelvis_height()])
        feet_height = self.feet_height()
        qpos = self.qpos_without_x()
        qvel = self.qvel()

        muscle_activations = self.muscle_activations()
        muscle_forces = self.muscle_forces()
        muscle_lengths = self.muscle_lengths()
        muscle_velocities = self.muscle_velocities()

        horizontal_velocity = np.array([self.horizontal_velocity()])

        observation = np.concatenate(
            (
                head_height,         #1
                pelvis_height,       #1   
                feet_height,         #2
                qpos,                #55
                qvel,                #56
                muscle_activations,  #120
                muscle_forces,       #120
                muscle_lengths,      #120
                muscle_velocities,   #120
                horizontal_velocity, #1
            )
        )

        return observation

    # In order to generate observation space automatically, this method cannot use class variable,
    # so it is defined as class method.
    @staticmethod
    def _get_obs_core(data):
        # just make sure the dimension is correct

        head_height = np.zeros(1)
        pelvis_height = np.zeros(1)
        feet_height = np.zeros(2)
        qpos = np.zeros(55)
        qvel = np.zeros(56)

        muscle_activations= np.zeros(120)
        muscle_forces = np.zeros(120)
        muscle_lengths = np.zeros(120)
        muscle_velocities = np.zeros(120)

        horizontal_velocity = np.zeros(1)

        observation = np.concatenate(
            (
                head_height,         #1
                pelvis_height,       #1   
                feet_height,         #2
                qpos,                #55
                qvel,                #56
                muscle_activations,  #120
                muscle_forces,       #120
                muscle_lengths,      #120
                muscle_velocities,   #120
                horizontal_velocity, #1
            )
        )
        return observation

    def step(self, action):

        self.do_simulation(action, self.frame_skip)
        observation = self.get_obs()
        
        reward = self.horizontal_velocity()
        
        terminated = self.terminated
        
        info = {
            "reward": reward,
        }

        self.steps += 1

        return observation, reward, terminated, False, info

    def reset_model(self):

        self.steps = 0

        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nv)

        self.set_state(qpos, qvel)

        observation = self.get_obs()

        return observation

    def viewer_setup(self):
        assert self.viewer is not None
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

    def render(self):
        return self.mujoco_renderer.render(
            self.render_mode, 
        )

    def _get_done(self):
        if self.head_height() < 0.9:
            return True
        if self.pelvis_height() < 0.6:
            return True
        if self.torso_angle() < -0.8 or self.torso_angle() > 0.8:
            return True
        return False

    def qpos_without_x(self):
        return self.data.qpos.copy()[1:]

    def qvel(self):
        return np.clip(self.data.qvel, -100, 100)

    def pelvis_height(self):
        pelvis_id = self.geom_name_list.index('pelvis')
        return self.data.geom_xpos[pelvis_id][2].copy()

    def feet_height(self):
        foot_id_r = self.body_name_list.index('r_pes')
        foot_id_l = self.body_name_list.index('l_pes')
        return np.array([self.data.xpos[foot_id_r][2].copy(),
                         self.data.xpos[foot_id_l][2].copy()])

    def head_height(self):
        head_id = self.body_name_list.index('head')
        return self.data.xpos[head_id][2].copy()

    def muscle_lengths(self):
        return self.data.actuator_length.copy()

    def muscle_velocities(self):
        return np.clip(self.data.actuator_velocity, -100, 100)

    def muscle_activations(self):
        return np.clip(self.data.act, -100, 100)

    def muscle_forces(self):
        return np.clip(self.data.actuator_force / 1000, -100, 100)

    def torso_angle(self):
        return self.data.qpos[4]

    def horizontal_velocity(self):
        # print(self.sensor_name_list)
        # print(self.data.sensordata)
        #sensor_name_list = [self.data.sensordata(ss_id).name for ss_id in range(self.model.nsensordata)]
        #torso_id = sensor_name_list.index('torso_subtreelinvel')
        return self.data.sensordata[0].copy()
