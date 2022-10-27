import random
from tqdm import tqdm
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential, Conv2d, MaxPool2d

import sys
sys.path.append('/Users/naminan/Development/Project/code')
from utils.ReplayBuffer import ReplayBuffer


class PPO(nn.Module):
    def __init__(self, num_class, env_type, learning_rate, gamma, lmbda, eps_clip, batch_size, device):
        super(PPO, self).__init__()

        self.env_type = env_type
        self.learning_rate = learning_rate

        self.data = []
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps_clip = eps_clip
        self.batch_size = batch_size

        self.device = device
    
        if self.env_type == 'Bioniceye':
            self.cnn_num_block = Sequential(
                Conv2d(in_channels=1, out_channels=32, kernel_size=7, stride=1, padding=0),
                ReLU(inplace=True), # perform the operation w/ using any additional memory
                MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False),

                Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=0),
                ReLU(inplace=True),
                MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False),

                Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),
                ReLU(inplace=True),
                MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False),

                Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0),
                ReLU(inplace=True)).to(self.device)
                
            self.linear_num_block_pi = Sequential(
                Linear(in_features=256*11*11, out_features=128, bias=True), 
                ReLU(inplace=True), 
                Linear(in_features=128, out_features=num_class, bias=True)).to(self.device)
            
            self.linear_num_block_v = Sequential(
                Linear(in_features=256*11*11, out_features=128, bias=True), 
                ReLU(inplace=True), 
                Linear(in_features=128, out_features=1, bias=True)).to(self.device)
        elif self.env_type == 'CartPole-v1':
            self.fc1   = Linear(4,256).to(self.device)
            self.fc_pi = Linear(256,2).to(self.device)
            self.fc_v  = Linear(256,1).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward_pi(self, xb):
        xb = xb.to(self.device)
        if len(xb.shape) == 3:
            xb = torch.unsqueeze(xb, 0) # -> (1, 1, 128, 128)

        if self.env_type == 'Bioniceye':
            xb = self.cnn_num_block(xb) # -> (batch_size, 256, 11, 11)
            xb = xb.view(xb.size(0), -1) # -> (batch_size, 256*11*11)
            xb = self.linear_num_block_pi(xb) # -> (batch_size, num_class)
        elif self.env_type == 'CartPole-v1':
            xb = F.relu(self.fc1(xb))
            xb = self.fc_pi(xb)
        out = F.softmax(xb, dim=-1) 
        return out
    
    def forward_v(self, xb):
        xb = xb.to(self.device)
        if len(xb.shape) == 3:
            xb = torch.unsqueeze(xb, 0) # -> (1, 1, 128, 128)

        if self.env_type == 'Bioniceye':   
            xb = self.cnn_num_block(xb)
            xb = xb.view(xb.size(0), -1) 
            xb = self.linear_num_block_v(xb)
        elif self.env_type == 'CartPole-v1':
            xb = F.relu(self.fc1(xb))
            xb = self.fc_v(xb)
        out = xb 
        return out
    
    def put_data(self, transition):
        self.data.append(transition)
    
    def concatenate_data(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        
        for (i, transition) in tqdm(enumerate(self.data), leave=False):
            s, a, r, s_prime, prob_a, done = transition          
            s, s_prime = np.expand_dims(s, 0), np.expand_dims(s_prime, 0) # batch

            if i == 0:
                state = s
                next_state = s_prime
            else:
                state = np.concatenate((state, s), axis=0)
                next_state = np.concatenate((next_state, s_prime), axis=0)
            state, next_state = torch.from_numpy(state), torch.from_numpy(next_state)

            a_lst.append([a])
            r_lst.append([r])
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            action, reward, prob_action, done_mask = \
            torch.tensor(a_lst), \
                torch.tensor(r_lst), \
                        torch.tensor(prob_a_lst), \
                            torch.tensor(done_lst, dtype=torch.float)       

        self.data = []
        return state, action, reward, next_state, prob_action, done_mask

    def train_net(self):
        full_state, full_action, full_reward, full_next_state, full_prob_action, full_done_mask = self.concatenate_data()
        if self.env_type == 'CartPole-v1':
            state, action, reward, next_state, prob_action, done_mask = full_state, full_action, full_reward, full_next_state, full_prob_action, full_done_mask
            trial_num = range(3)
        elif self.env_type == 'Bioniceye':
            trial_range = range(0, full_state.shape[0] // self.batch_size, self.batch_size)
        print(full_state.shape[0] // self.batch_size)
        for i in tqdm(trial_range):
            print(i, 'hi')
            if self.env_type == 'Bioniceye':
                state, action, reward, next_state, done_mask, prob_action = full_state[i:i+self.batch_size], full_action[i:i+self.batch_size], full_reward[i:i+self.batch_size], full_next_state[i:i+self.batch_size], full_done_mask[i:i+self.batch_size], full_prob_action[i:i+self.batch_size]
            
            td_target = reward.to(self.device) + self.gamma * self.forward_v(next_state) * done_mask.to(self.device)
            delta = td_target.to(self.device) - self.forward_v(state)
            delta = delta.detach().cpu().numpy()

            adv_lst = []
            adv = 0.0
            for delta_t in tqdm(delta[::-1], leave=False):
                adv = self.gamma * self.lmbda * adv + delta_t[0]
                adv_lst.append([adv])
            adv_lst.reverse()
            adv = torch.tensor(adv_lst, dtype=torch.float).to(self.device)

            pi = self.forward_pi(state)
            pi_action = pi.gather(1, action.to(self.device))
            ratio = torch.exp(torch.log(pi_action) - torch.log(prob_action.to(self.device))) # a / b == exp(log(a) - log(b))
            print(ratio.mean())
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * adv
            loss = - torch.min(surr1, surr2) + F.smooth_l1_loss(self.forward_v(state), td_target.detach())
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()


class Policy(nn.Module):
    def __init__(self, num_class, env_type, gamma, device):
        super(Policy, self).__init__()

        self.env_type = env_type
        self.device = device

        self.data = []
        self.gamma = gamma
        
        self.cnn_num_block = Sequential(
            Conv2d(in_channels=1, out_channels=32, kernel_size=7, stride=1, padding=0),
            ReLU(inplace=True), # perform the operation w/ using any additional memory
            MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False),

            Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=0),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False),

            Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False),

            Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0),
            ReLU(inplace=True)).to(self.device)
            
        self.linear_num_block_pi = Sequential(
            Linear(in_features=256*11*11, out_features=128, bias=True), 
            ReLU(inplace=True), 
            Linear(in_features=128, out_features=num_class, bias=True)).to(self.device)      

    def forward_pi(self, xb):
        xb = xb.to(self.device)
        if len(xb.shape) == 3:
            xb = torch.unsqueeze(xb, 0) # (1, 1, 128, 128)
        xb = self.cnn_num_block(xb)
        xb = xb.view(xb.size(0), -1) 
        xb = self.linear_num_block_pi(xb)
        out = F.softmax(xb, dim=-1) 
        return out
    
    def put_data(self, transition):
        self.data.append(transition)
    
    def train_net(self, optimizer):
        R = 0
        optimizer.zero_grad()
        for (reward, prob) in tqdm(self.data[::-1], leave=False):
            R = reward + R * self.gamma
            loss = - torch.log(prob) * R
            loss.requires_grad_(True)
            loss.backward()
        optimizer.step()
        self.data = []


class AC(nn.Module):
    def __init__(self, num_class, env_type, gamma, batch_size, device):
        super(AC, self).__init__()

        self.env_type = env_type
        self.device = device

        self.data = []
        self.gamma = gamma
        self.batch_size = batch_size
        
        self.cnn_num_block = Sequential(
            Conv2d(in_channels=1, out_channels=32, kernel_size=7, stride=1, padding=0),
            ReLU(inplace=True), # perform th eoperation w/ using any additional memory
            MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False),

            Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=0),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False),

            Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False),

            Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0),
            ReLU(inplace=True)).to(self.device)
            
        self.linear_num_block_pi = Sequential(
            Linear(in_features=256*11*11, out_features=128, bias=True), 
            ReLU(inplace=True), 
            Linear(in_features=128, out_features=num_class, bias=True)).to(self.device)
        
        self.linear_num_block_v = Sequential(
            Linear(in_features=256*11*11, out_features=128, bias=True), 
            ReLU(inplace=True), 
            Linear(in_features=128, out_features=1, bias=True)).to(self.device)

    def forward_pi(self, xb):
        xb = xb.to(self.device)
        if len(xb.shape) == 3:
            xb = torch.unsqueeze(xb, 0) # -> (1, 1, 128, 128)
        xb = self.cnn_num_block(xb) # -> (batch_size, 256, 11, 11)
        xb = xb.view(xb.size(0), -1) # -> (batch_size, 256*11*11)
        xb = self.linear_num_block_pi(xb) # -> (batch_size, num_class)
        out = F.softmax(xb, dim=-1) 
        return out
    
    def forward_v(self, xb):
        xb = xb.to(self.device)
        if len(xb.shape) == 3:
            xb = torch.unsqueeze(xb, 0) # -> (1, 1, 128, 128)
        xb = self.cnn_num_block(xb)
        xb = xb.view(xb.size(0), -1) 
        xb = self.linear_num_block_v(xb)
        out = xb 
        return out
    
    def put_data(self, transition):
        self.data.append(transition)
    
    def concatenate_data(self):
        a_lst, r_lst, done_lst, prob_a_lst = [], [], [], []
        
        for (i, transition) in tqdm(enumerate(self.data), leave=False):
            s, a, r, s_prime, done = transition
            s, s_prime = np.expand_dims(s, 0), np.expand_dims(s_prime, 0) # batch

            if i == 0:
                state = s
                next_state = s_prime
            else:
                state = np.concatenate((state, s), axis=0)
                next_state = np.concatenate((next_state, s_prime), axis=0)

            a_lst.append([a])
            r_lst.append([r])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

            action, reward, done_mask = torch.tensor(a_lst), \
                torch.tensor(r_lst), \
                    torch.tensor(done_lst)
            state, next_state = torch.from_numpy(state), torch.from_numpy(next_state)

        self.data = []
        return state, action, reward, next_state, done_mask

    def train_net(self, optimizer):
        full_state, full_action, full_reward, full_next_state, full_done_mask = self.concatenate_data()
        
        for i in tqdm(range(0, full_state.shape[0] // self.batch_size, self.batch_size), leave=False):
            state, action, reward, next_state, done_mask = full_state[i:i+self.batch_size], full_action[i:i+self.batch_size], full_reward[i:i+self.batch_size], full_next_state[i:i+self.batch_size], full_done_mask[i:i+self.batch_size]

            td_target = reward.to(self.device) + self.gamma * self.forward_v(next_state) * done_mask.to(self.device)
            delta = td_target.to(self.device) - self.forward_v(state)

            pi = self.forward_pi(state)
            pi_action = pi.gather(1, action.to(self.device))
            loss = - torch.log(pi_action) * delta.detach() + F.smooth_l1_loss(self.forward_v(state), td_target.detach())

            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()


class DQN(nn.Module):
    def __init__(self, num_class, env_type, gamma, batch_size, device):
        super(DQN, self).__init__()

        self.env_type = env_type
        self.device = device

        self.data = []
        self.gamma = gamma
        self.batch_size = batch_size
        
        self.cnn_num_block = Sequential(
            Conv2d(in_channels=1, out_channels=32, kernel_size=7, stride=1, padding=0),
            ReLU(inplace=True), # perform th eoperation w/ using any additional memory
            MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False),

            Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=0),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False),

            Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False),

            Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0),
            ReLU(inplace=True)).to(self.device)
            
        self.linear_num_block_pi = Sequential(
            Linear(in_features=256*11*11, out_features=128, bias=True), 
            ReLU(inplace=True), 
            Linear(in_features=128, out_features=num_class, bias=True)).to(self.device)

    def forward_pi(self, xb):
        xb = xb.to(self.device)
        if len(xb.shape) == 3:
            xb = torch.unsqueeze(xb, 0) # -> (1, 1, 128, 128)
        xb = self.cnn_num_block(xb)
        xb = xb.view(xb.size(0), -1) 
        xb = self.linear_num_block_pi(xb)
        out = F.softmax(xb, dim=-1) # (batch_size, num_class)
        return out

    def sample_action(self, state, epsilon):
        out = self.forward_pi(state)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 1)
        else:
            return out.argmax().item()
    
    def put_data(self, transition):
        self.data.append(transition)
    
    def concatenate_data(self):
        a_lst, r_lst, done_lst, prob_a_lst = [], [], [], []
        
        for (i, transition) in tqdm(enumerate(self.data), leave=False):
            s, a, r, s_prime, done = transition
            s, s_prime = np.expand_dims(s, 0), np.expand_dims(s_prime, 0) # batch

            if i == 0:
                state = s
                next_state = s_prime
            else:
                state = np.concatenate((state, s), axis=0)
                next_state = np.concatenate((next_state, s_prime), axis=0)

            a_lst.append([a])
            r_lst.append([r])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

            action, reward, done_mask = torch.tensor(a_lst), \
                torch.tensor(r_lst), \
                    torch.tensor(done_lst)
            state, next_state = torch.from_numpy(state), torch.from_numpy(next_state)

        self.data = []
        return state, action, reward, next_state, done_mask

    def train_net(self, optimizer, model_target):
        full_state, full_action, full_reward, full_next_state, full_done_mask = self.concatenate_data()
        
        for i in tqdm(range(0, full_state.shape[0] // self.batch_size, self.batch_size), leave=False):
            state, action, reward, next_state, done_mask = full_state[i:i+self.batch_size], full_action[i:i+self.batch_size], full_reward[i:i+self.batch_size], full_next_state[i:i+self.batch_size], full_done_mask[i:i+self.batch_size]

            q_out = self.forward_pi(state)
            q_a = q_out.gather(1, action.to(self.device))
            max_q_prime = model_target.forward_pi(next_state).max(1)[0].unsqueeze(1)
            target = reward.to(self.device) + self.gamma * max_q_prime * done_mask.to(self.device)
            loss = F.smooth_l1_loss(q_a, target)

            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
        
        return model_target