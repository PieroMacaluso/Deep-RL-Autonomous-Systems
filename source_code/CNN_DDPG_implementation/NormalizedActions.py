import gym
import numpy as np


class NormalizedActions(gym.ActionWrapper):
    """
    OpenAI Gym Wrapper to normalize the action in the range of the specific environment
    """
    
    def action(self, action):
        """
        :param action: action to be normalized
        :return: action normalized
        """
        low_bound = self.action_space.low
        upper_bound = self.action_space.high

        action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound)
        action = np.clip(action, low_bound, upper_bound)

        return action
