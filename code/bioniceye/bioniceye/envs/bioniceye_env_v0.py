
import os
import random
from glob import glob
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2 as cv
from gym import Env, spaces
import torch
from torch.utils.data import DataLoader

import sys
sys.path.append('/Users/naminan/Development/Project/code')
from dataloader.human720 import HumanDataLoader


class BionicEyeEnv_v0(Env):

    metadata = {'render_modes': ['human']}

    def __init__(self, image_dir, label_path, pred_dir, stim_type, top1, data_path, class_num, **kwargs):
        super(BionicEyeEnv_v0, self).__init__()

        """
        data_path: '/Users/naminan/Development/Project/code/data/210827_ANNA_Removing_uncontaminated_data.csv'
        """

        self.image_dir = image_dir
        self.label_path = label_path
        self.pred_dir = pred_dir
        self.stim_type = stim_type
        self.top1 = top1

        self.data_path = data_path
        self.class_num = class_num
        
        self.HumanDataLoader = HumanDataLoader(self.image_dir, self.label_path, self.pred_dir, self.stim_type, self.top1)
        self.trial_num = len(self.HumanDataLoader)
        self.batch_size = kwargs.get('batch_size', self.trial_num)
        self.trainloader = DataLoader(self.HumanDataLoader, batch_size=self.batch_size, num_workers=0, pin_memory=True)
        self.obs, self.label, self.human_pred = next(iter(self.trainloader))
        
        self.df = pd.read_csv(data_path)
        random.seed(22)
        self.l = list(range(self.df.shape[0]))
        self.set_1 = random.sample(self.l, self.class_num)
        self.old_unique_labels = list(set([self.df.iloc[i, 0] for i in self.set_1]))

    def reset(self):       
      """
      Return the first observation
      """
      return np.expand_dims(self.obs[0], 0), self.trial_num

    def step(self, t, action):
      """
      The next state does not depend on the current state and the action.
      The reward quantifies how much the agent's action aligns with the human's decision.
      End the episode if all the observations were seen by the agent.
      """
      obs, label, hum_pred = self.obs[t], self.label[t], self.human_pred[t]

      mach_pred = self.old_unique_labels[action]

      if (mach_pred == label) and (hum_pred == label):
        reward = 1
        info = True
      elif (mach_pred != label) and (hum_pred != label):
        reward = 0
        info = False
      if (mach_pred == label) and (hum_pred != label):
        reward = -1
        info = False
      elif (mach_pred != label) and (hum_pred == label):
        reward = -1
        info = False

      if info or t == self.trial_num - 1:
        done = True
      else:
        done = False

      return np.expand_dims(obs, 0), reward, done, info

    def render(self, obs, mode='plt'):
      assert mode in ['cv', 'plt'], "Invalid mode, must be either cv or plt"
      if mode == 'cv':
        cv.imshow(f'Low-Resolution Human Face', obs)
        cv.waitKey(10)
      
      elif mode == 'plt':
        plt.imshow(obs.numpy(), cmap='gray')
        plt.title(f'Low-Resolution Human Face')
        plt.show()
        
    def close(self):
        cv.destroyAllWindows()
