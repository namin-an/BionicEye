import time
from collections import deque
from tqdm import tqdm
import numpy as np
import torch
from torch.distributions import Categorical
import torch.optim as optim
import gym

import sys
sys.path.append('/Users/naminan/Development/Project/code')
from bioniceye.bioniceye.envs.bioniceye_env_v0 import BionicEyeEnv_v0
from models import PPO, Policy, AC, DQN, train_DQN
from utils.ReplayBuffer import ReplayBuffer


class Experiment():
    def __init__(self, image_dir, label_path, pred_dir, stim_type, top1, data_path, class_num, env_type, model_type, episode_num, print_interval, learning_rate, gamma, batch_size, render, device, *argv):

        self.image_dir = image_dir
        self.label_path = label_path
        self.pred_dir = pred_dir
        self.stim_type = stim_type
        self.top1 = top1

        self.data_path = data_path
        self.class_num = class_num

        self.env_type = env_type
        self.model_type = model_type
        self.episode_num = episode_num
        self.print_interval = print_interval
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.render = render

        self.device = device
        if len(argv[0]) >= 1:
            self.lmbda, self.eps_clip = argv[0][0], argv[0][1]

        if self.env_type == 'Bioniceye':
            self.env = BionicEyeEnv_v0(self.image_dir, self.label_path, self.pred_dir, self.stim_type, self.top1, self.data_path, self.class_num)
        elif self.env_type == 'CartPole-v1':
            self.env = gym.make('CartPole-v1')

        if self.model_type == 'PPO':
            self.model = PPO(self.class_num, self.env_type, self.learning_rate, self.gamma, self.lmbda, self.eps_clip, self.batch_size, self.device)
        elif self.model_type == 'REINFORCE':
            self.model = Policy(self.class_num, self.env_type, self.learning_rate, self.gamma, self.device)
        elif self.model_type == 'AC':
            self.model = AC(self.class_num, self.env_type, self.learning_rate, self.gamma, self.batch_size, self.device)
        elif self.model_type == 'DQN':
            self.model = DQN(self.class_num, self.env_type, self.device)
            self.model_target = DQN(self.class_num, self.env_type, self.device)
            self.model_target.load_state_dict(self.model.state_dict())
            self.buffer = ReplayBuffer()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self):
        train_returns = {i:[] for i in range(self.episode_num)} 
        train_score = 0.0

        for e in tqdm(range(self.episode_num)):
            if self.env_type == 'Bioniceye':
                state, trial_num = self.env.reset()
            elif self.env_type == 'CartPole-v1':
                state = self.env.reset()
                trial_num = 20
            done = False
            
            # collect experiences
            while not done:
                self.model.train()
                for t in tqdm(range(trial_num), leave=False):
                    if self.model_type == 'PPO' or self.model_type == 'REINFORCE' or self.model_type == 'AC':
                        probs = self.model.forward_pi(torch.from_numpy(state).float())
                    elif self.model_type == 'DQN':
                        epsilon = max(0.01, 0.08 - 0.01*(e/200)) #Linear annealing from 8% to 1%
                        action = self.model.sample_action(torch.from_numpy(state).float(), epsilon)                   
                    
                    if self.model_type != 'DQN':
                        m = Categorical(probs) 
                        action = m.sample()
                        if self.env_type == 'Bioniceye':
                            next_state, reward, done, _ = self.env.step(t, action.item())
                        elif self.env_type == 'CartPole-v1':
                            next_state, reward, done, _ = self.env.step(action.item())
                    else:
                        if self.env_type == 'Bioniceye':
                            next_state, reward, done, _ = self.env.step(t, action)
                        elif self.env_type == 'CartPole-v1':
                            next_state, reward, done, _ = self.env.step(action)
                    train_returns[e].append(reward)

                    if self.model_type == 'PPO':
                        if self.env_type == 'Bioniceye':
                            self.model.put_data((state, action, reward, next_state, probs[0][action].item(), done)) # probs: (1, self.class_num)
                        elif self.env_type == 'CartPole-v1':
                            self.model.put_data((state, action, reward/100., next_state, probs[action].item(), done))
                    elif self.model_type == 'REINFORCE':
                        if self.env_type == 'Bioniceye':
                            self.model.put_data((reward, probs[0][action]))
                        elif self.env_type == 'CartPole-v1':
                            self.model.put_data((reward/100., probs[action]))
                    elif self.model_type == 'AC':
                        if self.env_type == 'Bioniceye':
                            self.model.put_data((state, action, reward, next_state, done))
                        elif self.env_type == 'CartPole-v1':
                            self.model.put_data((state, action, reward/100., next_state, done))
                    elif self.model_type == 'DQN':
                        if self.env_type == 'Bioniceye':
                            self.buffer.put_data((state, action, reward, next_state, done))
                        elif self.env_type == 'CartPole-v1':
                            self.buffer.put_data((state, action, reward/100., next_state, done))
                    
                    state = next_state

                    if self.render:
                        self.env.render(state)
                    
                    train_score += reward
                    if done:
                        break

                    if e == 0:
                        best_model = self.model
                    else:
                        if sum(train_returns[e]) > sum(train_returns[e-1]):
                            best_model = self.model
        
            # train based on one set of collected experiences
            if self.model_type != 'DQN':
                self.model.train_net()
            else:
                if self.buffer.size() > 128:
                    self.model, self.model_target, self.optimizer = train_DQN(self.env_type, self.model, self.model_target, self.buffer, self.optimizer, self.learning_rate, self.gamma, self.batch_size, self.device)

            if e % self.print_interval == 0 and e != 0:
                print(f"EPISODE: {e} AVERAGE SCORE: {train_score/self.print_interval: .1f}")
                train_score = 0.0
                if self.model_type == 'DQN':
                    self.model_target.load_state_dict(self.model.state_dict())
            elif e == 0:
                print(f"EPISODE: {e} AVERAGE SCORE: {train_score: .1f}")
        
        if self.render:
            self.env.close()

        return best_model, train_returns