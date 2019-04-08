import cv2
import gym as gym
import numpy as np


class ImageWrapper(gym.ObservationWrapper):
    def __init__(self, image_size, *args):
        super(ImageWrapper, self).__init__(*args)
        assert isinstance(self.observation_space, gym.spaces.Box)
        old_space = self.observation_space
        self.image_size = image_size
        self.observation_space = gym.spaces.Box(self.observation(old_space.low), self.observation(old_space.high),
                                                dtype=np.float32)
    
    def observation(self, observation):
        # Resize image
        new_obs = cv2.resize(observation, (self.image_size, self.image_size))
        # Transform to a format Tensor-compatible
        new_obs = np.moveaxis(new_obs, 2, 0)
        return new_obs.astype(np.float32) / 255.0


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    env = ImageWrapper(96, gym.make("CarRacing-v0"))
    env.reset()
    for i in range(50):
        next_state, reward, done, _ = env.step(env.action_space.sample())
        # Transform from Tensor-compatible to image format to show
        next_state = np.moveaxis(next_state, 0, 2)
        plt.imshow(next_state)
        plt.show()
