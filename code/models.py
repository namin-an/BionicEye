import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential, Conv2d, MaxPool2d


class PPO(nn.Module):
    def __init__(self, class_num, env_type, device):
        super(PPO, self).__init__()

        self.env_type = env_type
        self.device = device
    
        if self.env_type == 'Bioniceye':
            self.cnn_num_block = Sequential(
                Conv2d(in_channels=1, out_channels=32, kernel_size=7, stride=1, padding=0),
                ReLU(inplace=True),
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
                Linear(in_features=128, out_features=class_num, bias=True)).to(self.device)
            
            self.linear_num_block_v = Sequential(
                Linear(in_features=256*11*11, out_features=128, bias=True), 
                ReLU(inplace=True), 
                Linear(in_features=128, out_features=1, bias=True)).to(self.device)

        elif self.env_type == 'CartPole-v1':
            self.fc1   = Linear(4,256).to(self.device)
            self.fc_pi = Linear(256,2).to(self.device)
            self.fc_v  = Linear(256,1).to(self.device)
    
    def forward(self, xb, **kwargs):
        xb = xb.to(self.device)
        try:
            if kwargs['pretrain']:
                xb = torch.unsqueeze(xb, 1)
        except:
            if len(xb.shape) == 3:
                xb = torch.unsqueeze(xb, 0)
        
        if self.env_type == 'Bioniceye':
            xb = self.cnn_num_block(xb) 
            xb = xb.view(xb.size(0), -1) 
            action_probs = self.linear_num_block_pi(xb)
            state_values = self.linear_num_block_v(xb)

        elif self.env_type == 'CartPole-v1':
            xb = F.relu(self.fc1(xb))
            action_probs = self.fc_pi(xb)
            state_values = self.fc_v(xb)
        
        try:
            if kwargs['pretrain']:
                action_probs = action_probs
        except:
            action_probs = F.softmax(action_probs, dim=-1)

        return action_probs, state_values

def train_PPO(env_type, model, memory, optimizer, gamma, lmbda, eps_clip, batch_size, device):
    if env_type == 'CartPole-v1':
        trial_range = range(10)
    elif env_type == 'Bioniceye':
        trial_range = range((memory.size()//batch_size) * 10)

    for _ in tqdm(trial_range, leave=False):
        state, action, reward, next_state, prob_action, done_mask = memory.sample(batch_size)

        pi, q = model(state)
        _, q_prime = model(next_state)
        
        td_target = reward.to(device) + gamma * q_prime * done_mask.to(device)
        delta = td_target.to(device) - q
        delta = delta.detach().cpu().numpy()

        adv_lst = []
        adv = 0.0
        for delta_t in tqdm(delta[::-1], leave=False):
            adv = gamma * lmbda * adv + delta_t[0]
            adv_lst.append([adv])
        adv_lst.reverse()
        adv = torch.tensor(adv_lst, dtype=torch.float).to(device)

        pi_action = pi.gather(1, action.to(device))
        ratio = torch.exp(torch.log(pi_action) - torch.log(prob_action.to(device))) 
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * adv
        loss = - torch.min(surr1, surr2) + F.smooth_l1_loss(q, td_target.detach())
        
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

    return model, optimizer


class AC(nn.Module):
    def __init__(self, class_num, env_type, gamma, batch_size, device):
        super(AC, self).__init__()

        self.env_type = env_type
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device
        
        if self.env_type == 'Bioniceye':
            self.cnn_num_block = Sequential(
                Conv2d(in_channels=1, out_channels=32, kernel_size=7, stride=1, padding=0),
                ReLU(inplace=True), 
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
                Linear(in_features=128, out_features=class_num, bias=True)).to(self.device)
            
            self.linear_num_block_v = Sequential(
                Linear(in_features=256*11*11, out_features=128, bias=True), 
                ReLU(inplace=True), 
                Linear(in_features=128, out_features=1, bias=True)).to(self.device)

        elif self.env_type == 'CartPole-v1':
            self.fc1   = Linear(4,256).to(self.device)
            self.fc_pi = nn.Linear(256,2).to(self.device)
            self.fc_v = nn.Linear(256,1).to(self.device)
        
    def forward(self, xb, **kwargs):
        xb = xb.to(self.device)     
        try:
            if kwargs['pretrain']:
                xb = torch.unsqueeze(xb, 1)
        except:
            if len(xb.shape) == 3:
                xb = torch.unsqueeze(xb, 0) 
        
        if self.env_type == 'Bioniceye':
            xb = self.cnn_num_block(xb) 
            xb = xb.view(xb.size(0), -1) 
            action_probs = self.linear_num_block_pi(xb) 
            state_values = self.linear_num_block_v(xb)

        elif self.env_type == 'CartPole-v1':
            xb = F.relu(self.fc1(xb))
            action_probs = self.fc_pi(xb)
            state_values = self.fc_v(xb)

        try:
            if kwargs['pretrain']:
                action_probs = action_probs
        except:
            action_probs = F.softmax(action_probs, dim=-1) 

        return action_probs, state_values

def train_AC(env_type, model, memory, optimizer, gamma, batch_size, device):
    if env_type == 'CartPole-v1':
        trial_range = range(10)
    elif env_type == 'Bioniceye':
        trial_range = range((memory.size()//batch_size) * 10)

    for _ in tqdm(trial_range, leave=False):
        state, action, reward, next_state, done_mask = memory.sample(batch_size)

        pi, q = model(state)
        _, q_prime = model(next_state)
        td_target = reward.to(device) + gamma * q_prime * done_mask.to(device)
        delta = td_target.to(device) - q

        pi_action = pi.gather(1, action.to(device))
        loss = - torch.log(pi_action) * delta.detach() + F.smooth_l1_loss(q, td_target.detach())

        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()
    
    return model, optimizer


class DQN(nn.Module):
    def __init__(self, class_num, env_type, device):
        super(DQN, self).__init__()

        self.env_type = env_type
        self.device = device
        
        if self.env_type == 'Bioniceye':
            self.cnn_num_block = Sequential(
                Conv2d(in_channels=1, out_channels=32, kernel_size=7, stride=1, padding=0),
                ReLU(inplace=True), 
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
                Linear(in_features=128, out_features=class_num, bias=True)).to(self.device)
            
        elif self.env_type == 'CartPole-v1':
            self.fc1   = Linear(4, 128).to(self.device)
            self.fc2 = nn.Linear(128, 128).to(self.device)
            self.fc3 = nn.Linear(128, 2).to(self.device)
        
    def forward(self, xb, **kwargs):
        xb = xb.to(self.device)
        try:
            if kwargs['pretrain']:
                xb = torch.unsqueeze(xb, 1)
        except:
            if len(xb.shape) == 3:
                xb = torch.unsqueeze(xb, 0) 
        
        if self.env_type == 'Bioniceye':
            xb = self.cnn_num_block(xb)
            xb = xb.view(xb.size(0), -1) 
            xb = self.linear_num_block_pi(xb)

        elif self.env_type == 'CartPole-v1':
            xb = F.relu(self.fc1(xb))
            xb = F.relu(self.fc2(xb))
            xb = self.fc3(xb)
        out = xb
        return out

    def sample_action(self, state, epsilon):
        out = self.forward(state)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 1)
        else:
            return out.argmax().item()
    
def train_DQN(env_type, model, model_target, memory, optimizer, gamma, batch_size, device):
    if env_type == 'CartPole-v1':
        trial_range = range(10)
    elif env_type == 'Bioniceye':
        trial_range = range(0, memory.size(), batch_size)

    for _ in tqdm(trial_range, leave=False):
        state, action, reward, next_state, done_mask = memory.sample(batch_size)

        q_out = model(state)
        q_a = q_out.gather(1, action.to(device))
        max_q_prime = model_target(next_state).max(1)[0].unsqueeze(1)
        target = reward.to(device) + gamma * max_q_prime * done_mask.to(device)
        loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model, model_target, optimizer


class CNN(nn.Module):
    def __init__(self, class_num):
        super(CNN, self).__init__()
    
        self.cnn_num_block = Sequential(
            Conv2d(in_channels=1, out_channels=32, kernel_size=7, stride=1, padding=0),
            ReLU(inplace=True), 
            MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False),

            Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=0),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False),

            Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False),

            Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0),
            ReLU(inplace=True))
            
        self.linear_num_block = Sequential(
            Linear(in_features=256*11*11, out_features=128, bias=True), 
            ReLU(inplace=True), 
            Linear(in_features=128, out_features=class_num, bias=True))

    def forward(self, xb):
        xb = torch.unsqueeze(xb, 1) 
        xb = self.cnn_num_block(xb)
        xb = xb.view(xb.size(0), -1) 
        xb = self.linear_num_block(xb)
        out = xb
        return out