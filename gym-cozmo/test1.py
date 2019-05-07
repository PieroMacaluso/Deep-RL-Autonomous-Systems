import sys

import cozmo
import gym
import gym_cozmo


if __name__ == '__main__':
    gym_cozmo.register(
        id='CozmoDriver-v0',
        entry_point='gym_cozmo.envs:CozmoEnv')
    env = gym.make('CozmoDriver-v0')
    state = env.reset()
    done = False
    total_reward = 0.0
    ciao = False
    while True:
        while robot.is_picked_up:
            continue
        next_state, reward, done, _ = env.step([1, -1])
        if ciao:
            break
    env.close()
    print(total_reward)