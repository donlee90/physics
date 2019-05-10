import os

import numpy as np
from PIL import Image

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms


class Experience(Dataset):
    def __init__(self, data_dirs, obs_len=100, obs_dim=4):
        self.episodes = []

        # Load experience data from data_dir
        episode_dirs = []
        for data_dir in data_dirs:
            episode_dirs += [os.path.join(data_dir, e) for e in os.listdir(data_dir)]
        for episode_dir in episode_dirs:
            actions = np.loadtxt(os.path.join(episode_dir, 'actions.txt'))
            observations = np.loadtxt(os.path.join(episode_dir, 'observations.txt'))

            actions = actions[:obs_len-1]
            observations = observations[:obs_len, :obs_dim]
            episode = {'actions': actions, 'observations': observations}

            self.episodes.append(episode)

    def __getitem__(self, index):
        episode = self.episodes[index]
        actions = torch.Tensor(episode['actions'])
        obs = torch.Tensor(episode['observations'][:-1])
        obs_next = torch.Tensor(episode['observations'][1:])

        return obs, actions, obs_next

    def __len__(self):
        return len(self.episodes)