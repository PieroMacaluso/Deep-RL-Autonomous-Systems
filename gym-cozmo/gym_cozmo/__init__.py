from gym.envs.registration import register


def initialize(robot, img_size):
    register(
        id='CozmoDriver-v0',
        entry_point='gym_cozmo.envs:CozmoEnv',
        kwargs={'robot': robot, 'image_dim': img_size})
    register(
        id='CozmoMock-v0',
        entry_point='gym_cozmo.envs:CozmoMock',
        kwargs={'robot': robot, 'image_dim': img_size})
