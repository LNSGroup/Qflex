from gymnasium.envs.registration import register


try:
    from envs.ostrich.OstrichRun_v1 import OstrichRunEnvV1

    register(
        id="OstrichRun-v1",
        entry_point="envs:OstrichRunEnvV1",
        max_episode_steps=3000,
    )
except ImportError:
    pass

try:
    from envs.ms_700_walk.locomotion_v1 import LocomotionEnvV1
    register(
        id="MS700Locomotion-v1",
        entry_point="envs:LocomotionEnvV1",
        max_episode_steps=3000,
    )
except ImportError as e:
    print("Failed to import MS700LocomotionEnvV1. Make sure the required dependencies are installed.", e)
    pass

try:
    from envs.smpl_humanoid.SMPLHumanoidLoco_v1 import HumEnvModifiedV1

    register(
        id="SMPLHumanoidJump-v1",
        entry_point="envs:HumEnvModifiedV1",
        max_episode_steps=3000,
        kwargs={"task": "jump-2"},
    )
except ImportError:
    pass
