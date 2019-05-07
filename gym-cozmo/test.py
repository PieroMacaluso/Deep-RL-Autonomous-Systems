import sys

import cozmo
import gym
import gym_cozmo


def run(sdk_conn):
    robot = sdk_conn.wait_for_robot()
    # self.robot.world.image_annotator.add_annotator('robotState', RobotStateDisplay)
    robot.enable_device_imu(True, True, True)
    # robot.say_text("Hi!").wait_for_completed()
    
    # Turn on image receiving by the camera
    robot.camera.image_stream_enabled = True
    gym_cozmo.register(
        id='CozmoDriver-v0',
        entry_point='gym_cozmo.envs:CozmoEnv',
        kwargs={'robot': robot})
    env = gym.make('CozmoDriver-v0')
    state = env.reset()
    done = False
    total_reward = 0.0
    ciao = False
    for i in range(10):
        print(i)
        while True:
            next_state, reward, done, _ = env.step([1, -1])
            if done:
                break
    env.close()
    print(total_reward)


if __name__ == '__main__':
    cozmo.setup_basic_logging()
    try:
        cozmo.connect(run)
    except KeyboardInterrupt as e:
        pass
    except cozmo.ConnectionError as e:
        sys.exit("A connection error occurred: %s" % e)
