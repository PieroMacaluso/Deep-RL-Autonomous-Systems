from gym.envs.registration import register

register(
    id='CozmoDriver-v0',
    entry_point='gym_cozmo.envs:CozmoEnv',
)