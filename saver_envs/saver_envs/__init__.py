from gym.envs.registration import register

register(
    id='robotArmEnv-v0',
    entry_point='saver_envs.envs:RobotArmEnv',
)

register(
    id='dimChooserEnv-v0',
    entry_point='saver_envs.envs:DimChooserEnv',
)

register(
    id='angleEnv-v0',
    entry_point='saver_envs.envs:AngleEnv',
)
