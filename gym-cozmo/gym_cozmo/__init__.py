from gym.envs.registration import register


def initialize(robot, img_h, img_w):
    register(
        id='CozmoDriver-v0',
        entry_point='gym_cozmo.envs:CozmoEnv',
        kwargs={'robot': robot, 'img_h': img_h, 'img_w': img_w})
    register(
        id='CozmoMock-v0',
        entry_point='gym_cozmo.envs:CozmoMock',
        kwargs={'robot': robot, 'img_h': img_h, 'img_w': img_w})
