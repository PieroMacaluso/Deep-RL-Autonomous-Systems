import sys
import matplotlib.pyplot as plt
import time

import cozmo
import gym
import gym_cozmo


def run(sdk_conn):
    robot: cozmo.robot = sdk_conn.wait_for_robot()
    robot.enable_device_imu(True, True, True)
    
    # Turn on image receiving by the camera
    robot.camera.image_stream_enabled = True
    gym_cozmo.initialize(robot, 140, 320)
    env = gym.make('CozmoMock-v0')
    state = env.reset()
    done = False
    total_reward = 0.0
    next_state, reward, done, _ = env.step([1, 0])
    plt.imshow(next_state)
    plt.show()
    next_state, reward, done, _ = env.step([1, 0])
    next_state, reward, done, _ = env.step([1, 0])
    next_state, reward, done, _ = env.step([1, 0])
    next_state, reward, done, _ = env.step([1, 0])
    robot.stop_all_motors()
    # for i in range(1):
    #     while env.is_human_controlled():
    #         continue
    #     print(i)
    #     while True:
    #         next_state, reward, done, _ = env.step([1,-1])
    #         if done:
    #             break
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
