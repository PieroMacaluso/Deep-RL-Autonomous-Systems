import time

import cozmo
import cozmo.robot as rb
import cv2
import gym
import numpy as np
from cozmo.util import Angle
from gym import spaces
from gym.utils import seeding

from gym_cozmo.envs.remote_control import start

MAX_F_SPEED = 150
MAX_T_SPEED = 100


class CozmoEnv(gym.Env):
    '''
    Cozmo Environment Class
    '''
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, robot: cozmo.robot.Robot, img_h: int, img_w: int):
        """
        Initialisation of environment parameters.
        @param robot: structure of Cozmo Robot provided by its SDK
        @type robot: cozmo.robot.Robot
        @param img_h: image height
        @type img_h: int
        @param img_w: image_width
        @type img_w: int
        """
        # choice_time: time between an action and the next one
        # this value is equal to 1 / number_frame_second
        # number_frame_second is generally up to 15 as reported in Cozmo SDK
        self.choice_time = 1 / 15
        self.last_action = None
        self.seed()
        self.img_h = img_h
        self.img_w = img_w
        # last_time: it stores the time of the last action. It is useful to calculate the reward (mm)
        self.last_time = 0
        self.robot = robot
        self.rc, self.thread = start(self.robot)
        self.robot.set_robot_volume(0.1)
        self.reward = 0.0
        self.state = None
        self.lift = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.head = spaces.Box(low=rb.MIN_HEAD_ANGLE.degrees, high=rb.MAX_HEAD_ANGLE.degrees, shape=(1,),
                               dtype=np.float32)
        self.action_space = spaces.Box(np.array([0, -1]), np.array([+1, +1]), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.img_h, self.img_w), dtype=np.float32)
    
    def step(self, action: spaces.Box):
        """
        Make a step in the environment
        @param action: action selected by the current policy (outside)
        @type action: spaces.Box
        @return: the next state, the reward of the actions just made, the done flag, other information
        @rtype: image, number, boolean, any
        """
        now_time = time.time()
        step_reward = -1
        if action is not None:
            self.drive(action)
            step_reward = action[0] * MAX_F_SPEED * self.choice_time
            self.last_time = now_time
        
        # Wait for the action to be executed
        time.sleep(self.choice_time)
        
        self.state = self.get_image()
        
        # Only a human can interrupt an episode
        if self.rc.is_human_controlled():
            self.robot.stop_all_motors()
            done = True
            step_reward = 0
        else:
            done = False
        
        return self.state, step_reward, done, {}
    
    def reset(self):
        """
        Reset the current environment by returning the reset state
        @return: reset state
        @rtype: image
        """
        # self.say("New Episode!")
        self.start_position()
        self.reward = 0.0
        self.state = self.get_image()
        self.last_time = time.time()
        return self.state
    
    def seed(self, seed=None):
        """
        Setup the seed for the current environment
        @param seed: seed
        @type seed: int
        @return:
        @rtype:
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def render(self, mode='human', close=False):
        """
        Unused method
        @param mode:
        @type mode:
        @param close:
        @type close:
        @return:
        @rtype:
        """
        pass
    
    def set_lift_height(self, height):
        """
        Set the height of the lift of Cozmo
        @param height:
        @type height:
        @return:
        @rtype:
        """
        self.robot.set_lift_height(height).wait_for_completed()
    
    def set_head_angle(self, degrees):
        """
        Set the head position of Cozmo
        @param degrees:
        @type degrees:
        @return:
        @rtype:
        """
        angle = Angle(degrees=degrees)
        self.robot.set_head_angle(angle).wait_for_completed()
        pass
    
    def close(self):
        self.start_position()
    
    def start_position(self):
        self.set_head_angle(self.head.low + 2)
        self.set_lift_height(self.lift.high)
    
    def get_image(self):
        """
        Get image from Cozmo Camera and do the proper transformation
        @return: cozmo camera image
        @rtype: image
        """
        observation = self.robot.world.latest_image.raw_image
        
        while observation is None:
            observation = self.robot.world.latest_image.raw_image
        # Returned screen requested by gym is HWC. Transpose it into torch order (CHW).\
        # screen = self.env.render(mode='rgb_array')
        observation = observation.convert("L")
        # screen = observation
        # screen_height, screen_width = screen.shape
        screen = np.ascontiguousarray(observation, dtype=np.float32) / 255
        # plt.imshow(screen)
        # screen = screen[-140:, :]
        screen = cv2.resize(screen, (self.img_w, self.img_h))
        # screen = screen.transpose((2, 0, 1))
        return screen
    
    def drive(self, action: spaces.Box):
        """
        Formula to convert actions to values understan
        @param action:
        @type action:
        @return:
        @rtype:
        """
        l_wheel_speed = action[0] * MAX_F_SPEED + action[1] * MAX_T_SPEED
        r_wheel_speed = action[0] * MAX_F_SPEED - action[1] * MAX_T_SPEED
        
        self.robot.drive_wheels(l_wheel_speed, r_wheel_speed, l_wheel_speed * 4, r_wheel_speed * 4)
    
    def say(self, message):
        """
        Wrapper to make Cozmo say something. It is used in the project only to feedback different phases
        @param message: Message to make Cozmo say
        @type message: string
        @return: nothing
        @rtype: nothing
        """
        self.robot.say_text(message).wait_for_completed()
    
    def is_human_controlled(self):
        """
        Return true if the human has control, false otherwise
        @return: result of the operation
        @rtype:
        """
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
        self.robot.stop_all_motors()
