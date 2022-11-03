
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
# sys.path.append('/Users/naminan/Development/Project/code')
from dataloader.human720 import HumanDataLoader


class BionicEyeEnv_v0(Env):

    metadata = {'render_modes': ['human']}

    def __init__(self, image_dir, label_path, pred_dir, stim_type, top1, data_path, class_num):
        super(BionicEyeEnv_v0, self).__init__()
        
        HumanDataset = HumanDataLoader(image_dir, label_path, pred_dir, stim_type, top1)
        self.trial_num = len(HumanDataset)
        trainloader = DataLoader(HumanDataset, batch_size=self.trial_num, num_workers=0, pin_memory=True, shuffle=False)
        self.obs, self.label, self.human_pred = next(iter(trainloader))
        
        df = pd.read_csv(data_path)
        random.seed(22)
        l = list(range(df.shape[0]))
        set_1 = random.sample(l, class_num)
        self.old_unique_labels = list(set([df.iloc[i, 0] for i in set_1]))

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
      label, hum_pred = label.item(), hum_pred.item()

      mach_pred = self.old_unique_labels[action]
      
      if (mach_pred == label) and (hum_pred == label):
        reward = 1
        info = False
      elif (mach_pred != label) and (hum_pred != label):
        reward = 1
        info = False
      if (mach_pred == label) and (hum_pred != label):
        reward = -1
        info = True
      elif (mach_pred != label) and (hum_pred == label):
        reward = -1 
        info = True
      
      if reward < -10 or t == self.trial_num - 1:
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
