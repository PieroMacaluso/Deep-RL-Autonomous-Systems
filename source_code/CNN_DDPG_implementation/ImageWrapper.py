import datetime

import cv2
import gym as gym
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from tensorboardX import SummaryWriter


class ImageWrapper(gym.ObservationWrapper):
    def __init__(self, image_size, *args):
        super(ImageWrapper, self).__init__(*args)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        assert isinstance(self.observation_space, gym.spaces.Box)
        self.old_space = self.observation_space
        self.image_size = image_size
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(3, self.image_size, self.image_size),
                                                dtype=np.float32)
    
    def observation(self, observation):
        # Returned screen requested by gym is HWC. Transpose it into torch order (CHW).\
        screen = self.env.render(mode='rgb_array')
        # screen = observation
        _, screen_height, screen_width = screen.shape
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = cv2.resize(screen, (self.image_size, self.image_size)).transpose((2, 0, 1))
        return screen
        # Resize image
        # # Transform to a format Tensor-compatible
        # new_obs = np.moveaxis(new_obs, 2, 0)
        # return new_obs.astype(np.float32) / 255.0


if __name__ == '__main__':
    folder = '{}_ImageWrapper_{}/'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), "Pendulum-v0",)
    writer = SummaryWriter(log_dir='runs/' + folder)
    
    env = ImageWrapper(64, gym.make("Pendulum-v0"))
    state = env.reset()
    episode_batch = torch.tensor(state).unsqueeze(0)
    for i in range(199):
        next_state, reward, done, _ = env.step(env.action_space.sample())
        # Transform from Tensor-compatible to image format to show
        
        episode_batch = torch.cat((episode_batch, torch.tensor(next_state).unsqueeze(0)))
    
    writer.add_images('state', episode_batch)
    
    env.close()
