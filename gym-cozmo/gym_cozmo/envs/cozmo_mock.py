import time

import cozmo
import cozmo.robot as rb
import gym
import numpy as np
from cozmo.util import Angle
from gym import spaces
from gym.utils import seeding

from gym_cozmo.envs.remote_control import start

MAX_F_SPEED = 150
MAX_T_SPEED = 100


class CozmoMock(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, robot: cozmo.robot.Robot, image_dim):
        self.choice_time = 0.1
        self.last_action = None
        self.seed()
        self.image_dim = image_dim
        self.last_time = 0
        self.robot = robot
        self.rc, self.thread = start(self.robot)
        self.reward = 0.0
        self.state = None
        self.lift = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.head = spaces.Box(low=rb.MIN_HEAD_ANGLE.degrees, high=rb.MAX_HEAD_ANGLE.degrees, shape=(1,),
                               dtype=np.float32)
        self.action_space = spaces.Box(np.array([0, -1]), np.array([+1, +1]), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.image_dim, self.image_dim), dtype=np.float32)
    
    def step(self, action: spaces.Box):
        now_time = time.time()
        step_reward = -1
        if action is not None:
            self.drive(action)
            step_reward = action[0] * MAX_F_SPEED * self.choice_time
            self.last_time = now_time
        
        # Wait for the action to be executed
        time.sleep(self.choice_time)
        
        self.state = None
        while self.state is None:
            self.state = self.get_image()
        
        if self.rc.is_human_controlled():
            done = True
            step_reward = 0
        else:
            done = False
        
        return self.state, step_reward, done, {}
    
    def reset(self):
        self.start_position()
        self.reward = 0.0
        self.state = self.get_image()
        self.last_time = time.time()
        return self.state
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def render(self, mode='human', close=False):
        pass
    
    def set_lift_height(self, height):
        pass
    
    def set_head_angle(self, degrees):
        angle = Angle(degrees=degrees)
        pass
    
    def close(self):
        self.start_position()
    
    def start_position(self):
        pass
    
    def get_image(self):
        screen = np.zeros((self.image_dim, self.image_dim))
        return screen
    
    def drive(self, action):
        pass
    
    def say(self, message):
        pass
    
    def is_human_controlled(self):
        return self.rc.is_human_controlled()
    
    def is_forget_enabled(self):
        return_value = self.rc.is_episode_to_be_discarded()
        return return_value
    
    def is_save_and_close(self):
        return self.rc.is_save_and_close()
    
    def reset_forget(self):
        self.rc.reset_forget()
    
    def is_test_phase(self):
        return self.rc.test_phase
    
    def stop_all_motors(self):
        pass