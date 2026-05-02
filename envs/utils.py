import os
from warnings import warn

import numpy as np
from gymnasium import spaces

try:
    import mujoco
except ImportError:
    warn("MuJoCo not found. Please install MuJoCo from https://www.mujoco.org/downloads/.")

def action_obs_check(cls):
    low = cls.action_space.low
    high = cls.action_space.high
    if (low == high).any():
        raise ValueError("Action space has the same low and high value")

    low = cls.observation_space.low
    high = cls.observation_space.high
    if (low == high).any():
        raise ValueError("Observation space has the same low and high value")

def get_observation_space(xml_path, get_obs_fn, obs_kwargs=None):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    obs = get_obs_fn(data, **obs_kwargs if obs_kwargs is not None else {})
    assert obs.ndim == 1, "Observation must be 1D"

    observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs.shape[0],), dtype=np.float64)
    return observation_space

def get_render_fps(xml_path, skip_frames):
    model = mujoco.MjModel.from_xml_path(xml_path)
    timestep = model.opt.timestep

    return int(round(1.0 / timestep / skip_frames))

def euler2quat(euler):
    """ Convert Euler Angles to Quaternions """
    euler = np.asarray(euler, dtype=np.float64)
    assert euler.shape[-1] == 3, "Invalid shape euler {}".format(euler)

    ai, aj, ak = euler[..., 2] / 2, -euler[..., 1] / 2, euler[..., 0] / 2
    si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
    ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    quat = np.empty(euler.shape[:-1] + (4,), dtype=np.float64)
    quat[..., 0] = cj * cc + sj * ss
    quat[..., 3] = cj * sc - sj * cs
    quat[..., 2] = -(cj * ss + sj * cc)
    quat[..., 1] = cj * cs - sj * sc
    return quat

def joint_name_to_dof_index(all_joint_name_list, joint_name_list):
    """
    Convert joint name list to joint index list
    """
    joint_index_list = []
    for joint_name in joint_name_list:
        if joint_name in all_joint_name_list:
            joint_index_list.append(all_joint_name_list.index(joint_name))
        else:
            raise ValueError("Joint name {} not found in all joint name list".format(joint_name))
    return joint_index_list

def mass_center(model, data):
    mass = np.expand_dims(model.body_mass, axis=1)
    xpos = data.xipos
    return (np.sum(mass * xpos, axis=0) / np.sum(mass))[0:].copy()

# Automatically get the observation space from the get_obs_fn
def get_observation_space2(xml_path, get_obs_fn, obs_kwargs=None, use_actuator_filter=True):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    if use_actuator_filter:
        actuator_filter = get_actuator_group_filter(model)
        obs = get_obs_fn(data, actuator_filter, **obs_kwargs if obs_kwargs is not None else {})
    else:
        obs = get_obs_fn(data, **obs_kwargs if obs_kwargs is not None else {})
    # if obs type is dict
    if isinstance(obs, dict):
        observation_space = spaces.Dict({
            key: spaces.Box(low=-np.inf, high=np.inf, shape=value.shape, dtype=np.float64)
            for key, value in obs.items()
        })
    else:
        assert obs.ndim == 1, "Observation must be 1D array"
        print("shape of observation:",obs.shape)
        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs.shape[0],), dtype=np.float64)
    return observation_space

def euler2mat(euler):
    """ Convert Euler Angles to Rotation Matrix """
    euler = np.asarray(euler, dtype=np.float64)
    assert euler.shape[-1] == 3, "Invalid shaped euler {}".format(euler)

    ai, aj, ak = -euler[..., 2], -euler[..., 1], -euler[..., 0]
    si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
    ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    mat = np.empty(euler.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 2, 2] = cj * ck
    mat[..., 2, 1] = sj * sc - cs
    mat[..., 2, 0] = sj * cc + ss
    mat[..., 1, 2] = cj * sk
    mat[..., 1, 1] = sj * ss + cc
    mat[..., 1, 0] = sj * cs - sc
    mat[..., 0, 2] = -sj
    mat[..., 0, 1] = cj * si
    mat[..., 0, 0] = cj * ci
    return mat

def get_com_velocity(data, model):
    """
    Compute the horizontal speed of the center of mass.
    """
    mass = np.expand_dims(model.body_mass, -1)
    cvel = -data.cvel

    velocity = (np.sum(mass * cvel, 0) / np.sum(mass))[3:5]
    velocity[0] *= -1 # Invert the x velocity so that it is forward positive

    return velocity

# def euler2quat(euler):
#     """ Convert Euler Angles to Quaternions """
#     euler = np.asarray(euler, dtype=np.float64)
#     assert euler.shape[-1] == 3, "Invalid shape euler {}".format(euler)

#     ai, aj, ak = euler[..., 2] / 2, -euler[..., 1] / 2, euler[..., 0] / 2
#     si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
#     ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
#     cc, cs = ci * ck, ci * sk
#     sc, ss = si * ck, si * sk

#     quat = np.empty(euler.shape[:-1] + (4,), dtype=np.float64)
#     quat[..., 0] = cj * cc + sj * ss
#     quat[..., 3] = cj * sc - sj * cs
#     quat[..., 2] = -(cj * ss + sj * cc)
#     quat[..., 1] = cj * cs - sj * sc
#     return quat


def mat2euler(mat):
    """ Convert Rotation Matrix to Euler Angles """
    _FLOAT_EPS = np.finfo(np.float64).eps
    _EPS4 = _FLOAT_EPS * 4.0

    mat = np.asarray(mat, dtype=np.float64)
    assert mat.shape[-2:] == (3, 3), "Invalid shape matrix {}".format(mat)

    cy = np.sqrt(mat[..., 2, 2] * mat[..., 2, 2] + mat[..., 1, 2] * mat[..., 1, 2])
    condition = cy > _EPS4
    euler = np.empty(mat.shape[:-1], dtype=np.float64)
    euler[..., 2] = np.where(condition,
                             -np.arctan2(mat[..., 0, 1], mat[..., 0, 0]),
                             -np.arctan2(-mat[..., 1, 0], mat[..., 1, 1]))
    euler[..., 1] = np.where(condition,
                             -np.arctan2(-mat[..., 0, 2], cy),
                             -np.arctan2(-mat[..., 0, 2], cy))
    euler[..., 0] = np.where(condition,
                             -np.arctan2(mat[..., 1, 2], mat[..., 2, 2]),
                             0.0)
    return euler


def mat2quat(mat):
    """ Convert Rotation Matrix to Quaternion """
    mat = np.asarray(mat, dtype=np.float64)
    assert mat.shape[-2:] == (3, 3), "Invalid shape matrix {}".format(mat)

    Qxx, Qyx, Qzx = mat[..., 0, 0], mat[..., 0, 1], mat[..., 0, 2]
    Qxy, Qyy, Qzy = mat[..., 1, 0], mat[..., 1, 1], mat[..., 1, 2]
    Qxz, Qyz, Qzz = mat[..., 2, 0], mat[..., 2, 1], mat[..., 2, 2]
    # Fill only lower half of symmetric matrix
    K = np.zeros(mat.shape[:-2] + (4, 4), dtype=np.float64)
    K[..., 0, 0] = Qxx - Qyy - Qzz
    K[..., 1, 0] = Qyx + Qxy
    K[..., 1, 1] = Qyy - Qxx - Qzz
    K[..., 2, 0] = Qzx + Qxz
    K[..., 2, 1] = Qzy + Qyz
    K[..., 2, 2] = Qzz - Qxx - Qyy
    K[..., 3, 0] = Qyz - Qzy
    K[..., 3, 1] = Qzx - Qxz
    K[..., 3, 2] = Qxy - Qyx
    K[..., 3, 3] = Qxx + Qyy + Qzz
    K /= 3.0
    # TODO: vectorize this -- probably could be made faster
    q = np.empty(K.shape[:-2] + (4,))
    it = np.nditer(q[..., 0], flags=['multi_index'])
    while not it.finished:
        # Use Hermitian eigenvectors, values for speed
        vals, vecs = np.linalg.eigh(K[it.multi_index])
        # Select largest eigenvector, reorder to w,x,y,z quaternion
        q[it.multi_index] = vecs[[3, 0, 1, 2], np.argmax(vals)]
        # Prefer quaternion with positive w
        # (q * -1 corresponds to same rotation as q)
        if q[it.multi_index][0] < 0:
            q[it.multi_index] *= -1
        it.iternext()
    return q


def quat2euler(quat):
    """ Convert Quaternion to Euler Angles """
    return mat2euler(quat2mat(quat))


def quat2mat(quat):
    """ Convert Quaternion to Euler Angles """
    _FLOAT_EPS = np.finfo(np.float64).eps
    quat = np.asarray(quat, dtype=np.float64)
    assert quat.shape[-1] == 4, "Invalid shape quat {}".format(quat)

    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    Nq = np.sum(quat * quat, axis=-1)
    s = 2.0 / Nq
    X, Y, Z = x * s, y * s, z * s
    wX, wY, wZ = w * X, w * Y, w * Z
    xX, xY, xZ = x * X, x * Y, x * Z
    yY, yZ, zZ = y * Y, y * Z, z * Z

    mat = np.empty(quat.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 0, 0] = 1.0 - (yY + zZ)
    mat[..., 0, 1] = xY - wZ
    mat[..., 0, 2] = xZ + wY
    mat[..., 1, 0] = xY + wZ
    mat[..., 1, 1] = 1.0 - (xX + zZ)
    mat[..., 1, 2] = yZ - wX
    mat[..., 2, 0] = xZ - wY
    mat[..., 2, 1] = yZ + wX
    mat[..., 2, 2] = 1.0 - (xX + yY)
    return np.where((Nq > _FLOAT_EPS)[..., np.newaxis, np.newaxis], mat, np.eye(3))

def mulQuat(qa, qb):
    res = np.zeros(4)
    res[0] = qa[0]*qb[0] - qa[1]*qb[1] - qa[2]*qb[2] - qa[3]*qb[3]
    res[1] = qa[0]*qb[1] + qa[1]*qb[0] + qa[2]*qb[3] - qa[3]*qb[2]
    res[2] = qa[0]*qb[2] - qa[1]*qb[3] + qa[2]*qb[0] + qa[3]*qb[1]
    res[3] = qa[0]*qb[3] + qa[1]*qb[2] - qa[2]*qb[1] + qa[3]*qb[0]
    return res

def average_downsample_2d(arr, factor):
    """
    Average downsample a 2D array by a factor of `factor`.
    """
    arr = arr.reshape(arr.shape[0] // factor, factor, arr.shape[1] // factor, factor)
    return arr.mean(axis=(1, 3))

def get_actuator_group_filter(model):
    actuator_group = model.actuator_group.copy()
    # zero to one and one to zero
    return ~(actuator_group.astype(bool))

def synergy_to_name_list(model, synergy):
    actuator_filter = get_actuator_group_filter(model)
    muscle_name_list = [model.actuator(act_id).name for act_id in range(model.nu)]
    muscle_name_list = [muscle_name_list[i] for i in range(model.nu) if actuator_filter[i]]
    synergy_name_list = []
    for syn in synergy:
        syn_group = []
        for i in syn:
            syn_group.append(muscle_name_list[i])
        synergy_name_list.append(syn_group)
        print(syn_group)

    print(synergy_name_list)
    return synergy_name_list

def name_list_to_synergy(model, name_lists):
    actuator_filter = get_actuator_group_filter(model)
    muscle_name_list = [model.actuator(act_id).name for act_id in range(model.nu)]
    muscle_name_list = [muscle_name_list[i] for i in range(model.nu) if actuator_filter[i]]
    synergy = []
    for name_list in name_lists:
        syn_group = []
        for name in name_list:
            syn_group.append(muscle_name_list.index(name))
        synergy.append(syn_group)

    print(synergy)
    return synergy

def index_list_to_dof(model, index_list):
    joint_name_list = [model.joint(jnt_id).name for jnt_id in range(model.njnt)]
    # get the dof list
    dof_list = [joint_name_list[i] for i in index_list]
    print(dof_list)
    return dof_list