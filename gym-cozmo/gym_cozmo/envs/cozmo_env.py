import cozmo
import cozmo.robot as rb
import gym
import numpy as np
from cozmo.util import Angle
from gym import spaces

# Collect events until released
from gym_cozmo.envs.remote_control import start

STATE_H = 64
STATE_W = 64
MAX_F_SPEED = 150
MAX_T_SPEED = 100


class CozmoEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, robot: cozmo.robot.Robot):
        self.robot = robot
        self.rc = start(self.robot)
        self.robot.set_robot_volume(0.1)
        self.reward = 0.0
        self.state = None
        self.lift = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.head = spaces.Box(low=rb.MIN_HEAD_ANGLE.degrees, high=rb.MAX_HEAD_ANGLE.degrees, shape=(1,),
                               dtype=np.float32)
        self.action_space = spaces.Box(np.array([0, -1]), np.array([+1, +1]), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(STATE_H, STATE_W), dtype=np.float32)
        self.say("All set!")
    
    def step(self, action: spaces.Box):
        step_reward = -1
        if action is not None:
            self.drive(action)
            step_reward = 1 + 0.001 * action[0]
        
        self.state = self.get_image()
        
        if self.rc.is_human_controlled():
            self.robot.stop_all_motors()
            done = True
            step_reward = -10 - 0.01 * action[0]
        else:
            done = False
        
        return self.state, step_reward, done, {'undo': self.rc.is_episode_to_be_discarded()}
    
    def reset(self):
        self.say("New Episode!")
        self.start_position()
        self.reward = 0.0
        self.state = self.get_image()
        return self.state
    
    def render(self, mode='human', close=False):
        pass
    
    def set_lift_height(self, height):
        self.robot.set_lift_height(height).wait_for_completed()
    
    def set_head_angle(self, degrees):
        angle = Angle(degrees=degrees)
        self.robot.set_head_angle(angle).wait_for_completed()
        pass
    
    def close(self):
        self.start_position()
    
    def start_position(self):
        self.set_head_angle(-20)
        self.set_lift_height(self.lift.high)
    
    def get_image(self):
        observation = self.robot.world.latest_image.raw_image
        # Returned screen requested by gym is HWC. Transpose it into torch order (CHW).\
        # screen = self.env.render(mode='rgb_array')
        screen = observation.convert("L")
        # screen = observation
        # screen_height, screen_width = screen.shape
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        # screen = cv2.resize(screen, (self.image_size, self.image_size))
        return screen
    
    def drive(self, action):
        l_wheel_speed = action[0] * MAX_F_SPEED + action[1] * MAX_T_SPEED
        r_wheel_speed = action[0] * MAX_F_SPEED - action[1] * MAX_T_SPEED
        
        self.robot.drive_wheels(l_wheel_speed, r_wheel_speed, l_wheel_speed * 4, r_wheel_speed * 4)
    
    def say(self, message):
        self.robot.say_text(message).wait_for_completed()
    
    def is_human_controlled(self):
        return self.rc.is_human_controlled()
    
    def is_test_phase(self):
        return self.rc.test_phase;
