import time

import cozmo
import gym


class CozmoEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        self.thread = None
        self.robot = None
        self.cozmoEnabled = True
        self.active_viewer = False
        self.lists = []
        self.async_mode = 'threading'
        cozmo.run_program(self.cozmo_program, )
        self.robot.say_text("Hi!").wait_for_completed()
    
    def cozmo_program(self, _robot: cozmo.robot.Robot):
        self.robot = _robot
    
    def step(self, action):
        pass
    
    def reset(self):
        pass
    
    def render(self, mode='human', close=False):
        pass


def say_lets_go(robot: cozmo.robot.Robot):
    robot.say_text("Let's go!").wait_for_completed()


def go_straight(robot: cozmo.robot.Robot):
    robot.say_text("Let's go!").wait_for_completed()
