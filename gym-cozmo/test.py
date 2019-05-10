import sys

import cozmo
import gym
import gym_cozmo


def run(sdk_conn):
    robot = sdk_conn.wait_for_robot()
    robot.enable_device_imu(True, True, True)
    
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
    for i in range(10):
        while env.is_human_controlled():
            continue
        print(i)
        while True:
            next_state, reward, done, _ = env.step(env.action_space.sample())
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
