import gym


class NormalizedActions(gym.ActionWrapper):
    """
    OpenAI Gym Wrapper to normalize the action.
    """
    
    def action(self, action):
        """
        Transform the action normalized between [0,1] to the correct action-space bound of the environment.
        
        :param action: action normalized in [0, 1]
        :return: action de-normalized in the environment space
        """
        action = action * (self.action_space.high - self.action_space.low) + self.action_space.low
        return action

    def _reverse_action(self, action):
        """
        Normalize the action between [0, 1]

        :param action: action in the action-space range
        :return: action normalized in [0, 1]
        """
        action = (action - self.action_space.low) / (self.action_space.high - self.action_space.low)
        return action
