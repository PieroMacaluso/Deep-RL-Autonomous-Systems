from gym.envs.registration import register

register(
    id='CozmoWithoutRobot-v0',
    entry_point='gym_cozmo.envs:CozmoEnv'
)
