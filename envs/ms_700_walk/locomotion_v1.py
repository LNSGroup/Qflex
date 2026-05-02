import os
from typing import Dict, Any, Optional
import numpy as np
from gymnasium.utils import EzPickle
from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from envs.utils import get_render_fps, action_obs_check, joint_name_to_dof_index
from envs.ms_700_walk.trajectory import Trajectory

class LocomotionEnvV1(MujocoEnv, EzPickle):

    metadata: Dict[str, Any] = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 10,
    }
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "w_qpos": 50, 
        "w_qvel": 0.1, 
        "w_act": 1, 
        "w_vel": 5, 
        "w_healthy": 100
    }
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        skip_frames: int = 10,
        reset_noise_scale = 0.0,
        random_init = True,
        terminate_time = 6,
        target_y_vel = 0.0,
        target_x_vel = 1.25,
        model_type = "fullbody",
        reward_dict = DEFAULT_RWD_KEYS_AND_WEIGHTS,
        **kwargs
    ):

        if model_type == "fullbody":
            model_path = os.path.dirname(__file__) + "/MS-Human-700/MS-Human-700.xml"
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        model_path = os.path.abspath(model_path)
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]

        fps = get_render_fps(model_path, skip_frames)
        self.metadata["render_fps"] = fps
        self.control_timestep = 1 / fps

        EzPickle.__init__(
            self,
            render_mode,
            skip_frames,
            reset_noise_scale,
            **kwargs
        )
        
        self.render_mode = render_mode
        self._reset_noise_scale = reset_noise_scale
        self.reward_weight = reward_dict
        self.terminated_time = terminate_time
        self._reset_time = 0.
        self.target_y_vel = target_y_vel
        self.target_x_vel = target_x_vel
        self._random_init = random_init
        self.model_type = model_type

        self.init_trajectory()

        # Initialize MujocoEnv
        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64)
        MujocoEnv.__init__(
            self, model_path, skip_frames, observation_space=observation_space, render_mode=render_mode, camera_name="record_camera", **kwargs
        )

        self.init_qpos[:] = self.model.key_qpos[0].copy()
        self.body_name_list = [self.model.body(body_id).name for body_id in range(self.model.nbody)]
        self.joint_name_list = [self.model.joint(jnt_id).name for jnt_id in range(self.model.njnt)]
        self.muscle_name_list = [self.model.actuator(act_id).name for act_id in range(self.model.nu)]

        if self.model_type == "fullbody":
            self.dof_torso = joint_name_to_dof_index(self.joint_name_list, self.dof_torso_name)
            for side in ["r", "l"]:
                name_leg = [f"{joint_name}_{side}" for joint_name in self.dof_leg_name]
                name_arm = [f"{joint_name}_{side}" for joint_name in self.dof_arm_name]
                setattr(self, f"dof_leg_{side}", joint_name_to_dof_index(self.joint_name_list, name_leg))
                setattr(self, f"dof_arm_{side}", joint_name_to_dof_index(self.joint_name_list, name_arm))
            self.dof_joint_mask = np.concatenate((self.dof_leg_r, self.dof_leg_l, self.dof_torso, self.dof_arm_r, self.dof_arm_l))
        elif self.model_type == "legs":
            for side in ["r", "l"]:
                name_leg = [f"{joint_name}_{side}" for joint_name in self.dof_leg_name]
                setattr(self, f"dof_leg_{side}", joint_name_to_dof_index(self.joint_name_list, name_leg))
            self.dof_joint_mask = np.concatenate((self.dof_leg_r, self.dof_leg_l))
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self.qpos_target, self.qvel_target = self.trajectory.query(time=0.)
        observation, _ = self._get_obs()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(observation.shape[0],), dtype=np.float64)
        action_obs_check(self)

        print("observation space shape: ", self.observation_space.shape)
        print("action space shape: ", self.action_space.shape)
    
    def seed(self, seed=0):
        pass
            
    @property
    def is_healthy(self):
        if self.data.body('pelvis').xpos[2] < 0.6:
            return False
        if self.data.body('sternum').xpos[2] < 1:
            return False

        # Pitch
        if abs(self.data.qpos[3]) > 0.4:
            return False
        # Roll
        if abs(self.data.qpos[4]) > 0.4:
            return False
        # Yaw
        if abs(self.data.qpos[5]) > 0.4:
            return False
        
        if np.linalg.norm(self.qpos_dist) > 3:
            return False

        return True
    
    @property
    def terminated(self):
        terminated = (not self.is_healthy)
        if self.data.time > self.terminated_time:
            terminated = True
        return terminated
    
    def _get_obs(self):
        qpos = self.data.qpos.flat.copy() 
        qvel = self.data.qvel.flat.copy() 
        qacc = self.data.qacc.flat.copy()

        act = self.data.act.flat.copy()
        actuator_forces = self.data.actuator_force.flat.copy() / 1000
        actuator_forces = actuator_forces.clip(-100, 100)
        actuator_length = self.data.actuator_length.flat.copy()
        actuator_velocity = self.data.actuator_velocity.flat.copy().clip(-100, 100)

        sim_time = np.array([self.data.time])
        phase_var = np.array([self.data.time / self.trj_period_time % 1])
        pelvis_pos = self.data.body('pelvis').xpos.copy()
        sternum_pos = self.data.body('sternum').xpos.copy()
        self.qpos_dist = self.qpos_target[:self.model.nq] - qpos

        obs_dict = {
            "qpos": qpos,
            "qvel": qvel,
            "qacc": qacc,
            "act": act,
            "actuator_forces": actuator_forces,
            "actuator_length": actuator_length,
            "actuator_velocity": actuator_velocity,
            "sim_time": sim_time,
            "phase_var": phase_var,
            "pelvis_pos": pelvis_pos,
            "sternum_pos": sternum_pos,
            "qpos_dist": self.qpos_dist,
        }
        
        observation = np.concatenate([obs_dict[key] for key in obs_dict.keys()])

        return observation, obs_dict
    
    def step(self, action):
        
        self.qpos_target, self.qvel_target = self.trajectory.query(time=self.data.time + self._reset_time)

        self.do_simulation(action, self.frame_skip)
        observation, obs_dict = self._get_obs()

        # Compute reward
        vel_reward = self._get_vel_reward(self.target_y_vel, self.target_x_vel) * self.reward_weight["w_vel"]
        qpos_reward = self._get_qpos_reward() * self.reward_weight["w_qpos"]
        qvel_reward = self._get_qvel_reward() * self.reward_weight["w_qvel"]
        act_reward = self._get_act_reward() * self.reward_weight["w_act"]
        healthy_reward = self._get_healthy_reward() * self.reward_weight["w_healthy"]
        reward = vel_reward + qpos_reward + qvel_reward + act_reward + healthy_reward

        terminated = self.terminated
        truncated = False

        info = {
            "vel_reward": vel_reward,
            "qpos_reward": qpos_reward,
            "qvel_reward": qvel_reward,
            "act_reward": act_reward,
            "healthy_reward": healthy_reward,
            "total_reward": reward,
        }

        return observation, reward, terminated, truncated, info
    
    def reset_model(self):
        # Reset time
        self._reset_time = 0 if not self._random_init else float(self.np_random.random() * self.trj_period_time)
        
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        self.qpos_target, self.qvel_target = self.trajectory.query(time=self.data.time + self._reset_time)
        
        self.init_qpos = self.qpos_target[:self.model.nq]
        self.init_qvel = self.qvel_target[:self.model.nv]

        qpos = self.init_qpos + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nv)

        self.set_state(qpos, qvel)

        observation, obs_dict = self._get_obs()
        
        return observation
    
    def render(self, mode=None):
        return super().render()
    
    def init_trajectory(self):
        data_path = os.path.dirname(__file__)
        qpos_file_path = data_path + "/motion_data/walking_qpos.csv"
        self.trajectory = Trajectory(file_path=qpos_file_path)
        self.trj_period_time = self.trajectory.data_time

        self.dof_torso_name = ["L5_S1_FE", "L5_S1_LB", "L5_S1_AR", 
                               "T12_L1_FE", "T12_L1_LB", "T12_L1_AR", 
                               "T1_head_neck_FE", "T1_head_neck_LB", "T1_head_neck_AR"]
        self.dof_leg_name = ["hip_flexion", "hip_adduction", "hip_rotation", "knee_angle", "ankle_angle", "subtalar_angle", "mtp_angle"]
        self.dof_arm_name = ["elv_angle", "shoulder_elv", "shoulder_rot", "elbow_flexion", "pro_sup", "deviation", "flexion"]

    def _get_healthy_reward(self):
        return 1 if self.is_healthy else 0

    def _get_act_reward(self):
        return -np.linalg.norm(self.data.act) / self.model.na

    def _get_qvel_reward(self):
        return -np.linalg.norm(self.qvel_target[:self.model.nv] - self.data.qvel)

    def _get_qpos_reward(self):
        return np.sum(-np.square(self.qpos_dist[self.dof_joint_mask]))

    def _get_vel_reward(self, target_y_vel, target_x_vel):
        vel = self._get_com_velocity()
        return -np.square(target_y_vel - vel[1]) - np.square(target_x_vel - vel[0])

    def _get_com_velocity(self):
        mass = np.expand_dims(self.model.body_mass, -1)
        cvel = -self.data.cvel
        vel = (np.sum(mass * cvel, 0) / np.sum(mass))[3:5]
        vel[0] *= -1  # flip x axis
        return vel
