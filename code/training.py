import time
from collections import deque
from tqdm import tqdm
import numpy as np
import torch
from torch.distributions import Categorical
import torch.optim as optim

import sys
sys.path.append('/Users/naminan/Development/Project/code')
from bioniceye.bioniceye.envs.bioniceye_env_v0 import BionicEyeEnv_v0
from models import PPO, Policy, AC, DQN
from utils.average_meter import AverageMeter
from utils.early_stop import EarlyStopping
from utils.plot_figures import lineplot


class Experiment():
    def __init__(self, image_dir, label_path, pred_dir, stim_type, top1, data_path, class_num, device, check_time_usage, model_type, episode_num, print_interval, learning_rate, discount, average_window):

        self.image_dir = image_dir
        self.label_path = label_path
        self.pred_dir = pred_dir
        self.stim_type = stim_type
        self.top1 = top1

        self.data_path = data_path
        self.class_num = class_num

        self.device = device
        self.check_time_usage = check_time_usage
        self.model_type = model_type
        self.episode_num = episode_num
        self.print_interval = print_interval
        self.learning_rate = learning_rate
        self.discount = discount
        self.average_window = average_window

        self.env = BionicEyeEnv_v0(self.image_dir, self.label_path, self.pred_dir, self.stim_type, self.top1, self.data_path, self.class_num)
        if self.model_type == 'PPO':
            self.model = PPO(self.class_num, self.device)
        elif self.model_type == 'REINFORCE':
            self.model = Policy(self.class_num, self.device)
        elif self.model_type == 'AC':
            self.model = AC(self.class_num, self.device)
        elif self.model_type == 'DQN':
            self.model = DQN(self.class_num, self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self):
        train_returns = {i:[] for i in range(self.episode_num)} 
        test_returns = {i:[] for i in range(self.episode_num)}
        train_score, test_score = 0.0, 0.0

        if self.model_type == 'DQN':
            model_target = DQN(self.class_num, self.device)

        for e in tqdm(range(self.episode_num)):
            self.model.train()
            state, trial_num = self.env.reset()
            done = False

            # collect experiences
            for t in tqdm(range(trial_num), leave=False):
                if self.model_type == 'PPO' or self.model_type == 'REINFORCE' or self.model_type == 'AC':
                    probs = self.model.forward_pi(torch.from_numpy(state).float())
                elif self.model_type == 'DQN':
                    epsilon = max(0.01, 0.08 - 0.01*(e/200)) #Linear annealing from 8% to 1%
                    action = self.model.sample_action(torch.from_numpy(state).float(), epsilon)                   
                
                if self.model_type != 'DQN':
                    m = Categorical(probs) 
                    action = m.sample()
                    next_state, reward, done, finetune = self.env.step(t, action.item())
                else:
                    next_state, reward, done, finetune = self.env.step(t, action)
                train_returns[e].append(reward)

                if self.model_type == 'PPO':
                    self.model.put_data((state, action, reward, next_state, done, probs[0][action])) # probs: (1, self.class_num)
                elif self.model_type == 'REINFORCE':
                    self.model.put_data((reward, probs[0][action]))
                elif self.model_type == 'AC' or self.model_type == 'DQN':
                    self.model.put_data((state, action, reward, next_state, done))
                
                state = next_state

                # self.env.render(state)
                
                train_score += reward
                if done:
                    break

            # train based on one set of collected experiences
            if self.model_type != 'DQN':
                self.model.train_net(self.optimizer)
            else:
                model_target = self.model.train_net(self.optimizer, model_target)

            # validate the trained agent  
            with torch.inference_mode():
                self.model.eval()
                state, trial_num = self.env.reset()
                
                for t in tqdm(range(trial_num), leave=False):
                    if self.model_type == 'PPO' or self.model_type == 'REINFORCE' or self.model_type == 'AC':
                        probs = self.model.forward_pi(torch.from_numpy(state).float())
                    elif self.model_type == 'DQN':
                        epsilon = max(0.01, 0.08 - 0.01*(e/200)) #Linear annealing from 8% to 1%
                        action = self.model.sample_action(torch.from_numpy(state).float(), epsilon)                   
                    
                    if self.model_type != 'DQN':
                        m = Categorical(probs) 
                        action = m.sample()
                        next_state, reward, done, finetune = self.env.step(t, action.item())
                    else:
                        next_state, reward, done, finetune = self.env.step(t, action)
                    if finetune:
                        value = 1
                    else:
                        value = 0
                    test_returns[e].append(value)

                    if self.model_type == 'PPO':
                        self.model.put_data((state, action, reward, next_state, done, probs[0][action])) # probs: (1, self.class_num)
                    elif self.model_type == 'REINFORCE':
                        self.model.put_data((reward, probs[0][action]))
                    elif self.model_type == 'AC' or self.model_type == 'DQN':
                        self.model.put_data((state, action, reward, next_state, done))
                    
                    state = next_state
                    
                    test_score += reward
                    if done:
                        break

                if e == 0:
                    best_model = self.model
                else:
                    if sum(test_returns[e]) > sum(test_returns[e-1]):
                        best_model = self.model

            if e % self.print_interval == 0 and e != 0:
                print(f"EPISODE: {e} AVERAGE TRAINING SCORE: {train_score/self.print_interval: .1f} AVERAGE VALIDATION SCORE: {test_score/self.print_interval: .1f}")
                train_score, test_score = 0.0, 0.0
                if self.model_type == 'DQN':
                    model_target.load_state_dict(self.model.state_dict())
            elif e == 0:
                print(f"EPISODE: {e} AVERAGE TRAINING SCORE: {train_score: .1f} AVERAGE VALIDATION SCORE: {test_score: .1f}")
        
        # self.env.close()
        return best_model, train_returns, test_returns