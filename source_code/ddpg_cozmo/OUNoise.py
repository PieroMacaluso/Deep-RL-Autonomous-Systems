import numpy as np
from gym import spaces


class OUNoise(object):
    """
    OUNoise classe documentation.
    This is a simple implementation of the Ornstein Uhlenbeck process noise:
        x_(t+1) = x_t + theta * (mu - x_t) + sigma * e_t
    where theta, mu and sigma are hyper-parameters.
    """
    
    def __init__(self, action_space, mu=0.0, sigma=0.2, theta=0.15):
        """
        Constructor of the OU Noise

        :param action_space: action_ space of Gym Env
        :param mu: mu hyperparameter
        :param theta: theta hyperparameter
        :param max_sigma: max_sigma hyperparameter
        :param min_sigma: min_sigma hyperparameter
        :param decay_period: decay period
        """
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.state = np.ones(self.action_dim)
        self.reset()
    
    def reset(self):
        """
        Reset the OU Noise.
        """
        self.state = np.ones(self.action_dim) * self.mu
    
    def evolve_state(self) -> np.ndarray:
        """
        Evolve the state of OU Noise applying the transformation and returning the vector.
        :return: Current State of OU Noise
        """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state * 0.5
    
    def get_action(self, action_: np.ndarray, eps: float = 1.0) -> np.ndarray:
        """
         Return the action provided as parameter plus the calculated noise.

        :param action_: Action-Value vector with OU Noise applied
        :param eps: epsilon decay of the noise (optional and external)
        :return:  Action-Value vector with OU Noise applied
        """
        ou_state = self.evolve_state()
        return_action = action_ + eps * ou_state
        return np.clip(return_action, self.low, self.high)


if __name__ == '__main__':
    action_space = spaces.Box(np.array([0, -1]), np.array([+1, +1]), dtype=np.float32)
    
    ou = OUNoise(action_space)
    states = []
    for i in range(1000):
        states.append(ou.evolve_state())
    import matplotlib.pyplot as plt
    
    plt.plot(states)
    plt.show()
