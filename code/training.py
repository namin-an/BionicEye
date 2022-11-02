from tqdm import tqdm
import torch
from torch.distributions import Categorical
from torch.utils.data import DataLoader, random_split
import gym

import sys
sys.path.append('/Users/naminan/Development/Project/code')
from bioniceye.bioniceye.envs.bioniceye_env_v0 import BionicEyeEnv_v0
from dataloader.kface16000 import KFaceDataLoader
from models import PPO, Policy, AC, DQN, train_DQN, CNN
from utils.ReplayBuffer import ReplayBuffer
from utils.early_stop import EarlyStopping


class ExperimentRL():
    def __init__(self, image_dir, label_path, pred_dir, stim_type, top1, data_path, class_num, env_type, model_type, episode_num, print_interval, learning_rate, gamma, batch_size, render, device, *argv):

        self.env_type = env_type
        self.model_type = model_type
        self.episode_num = episode_num
        self.print_interval = print_interval
        self.gamma = gamma
        self.batch_size = batch_size
        self.render = render

        self.device = device
        if len(argv[0]) >= 1:
            self.lmbda, self.eps_clip = argv[0][0], argv[0][1]

        # Setup environments
        if self.env_type == 'Bioniceye':
            self.env = BionicEyeEnv_v0(image_dir, label_path, pred_dir, stim_type, top1, data_path, self.class_num)
        elif self.env_type == 'CartPole-v1':
            self.env = gym.make('CartPole-v1')

        # Prepare models
        if self.model_type == 'PPO':
            self.model = PPO(class_num, self.env_type, learning_rate, self.gamma, self.lmbda, self.eps_clip, self.batch_size, self.device)
        elif self.model_type == 'REINFORCE':
            self.model = Policy(class_num, self.env_type, learning_rate, self.gamma, self.device)
        elif self.model_type == 'AC':
            self.model = AC(class_num, self.env_type, learning_rate, self.gamma, self.batch_size, self.device)
        elif self.model_type == 'DQN':
            self.model = DQN(class_num, self.env_type, self.device)
            self.model_target = DQN(class_num, self.env_type, self.device)
            self.model_target.load_state_dict(self.model.state_dict())
            self.buffer = ReplayBuffer()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

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
            
            # Collect experiences
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
                            self.model.put_data((state, action, reward, next_state, probs[0][action].item(), done)) # probs: (1, class_num)
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
        
            # Train with collected transitions
            if self.model_type != 'DQN':
                self.model.train_net()
            else:
                if self.buffer.size() > 128:
                    self.model, self.model_target, self.optimizer = train_DQN(self.env_type, self.model, self.model_target, self.buffer, self.optimizer, self.gamma, self.batch_size, self.device)

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


class ExperimentSL():
    def __init__(self, image_dir, data_path, class_num, epoch_num, print_interval, learning_rate, batch_size, seed, model_file_path, device):

        self.epoch_num = epoch_num
        self.print_interval = print_interval

        self.device = device

        # Prepare dataloader
        KFaceDataset = KFaceDataLoader(image_dir, data_path, None, class_num, 'train', seed)
        total_size = len(KFaceDataset)
        train_size = int(total_size*0.8)
        trainDataset, validDataset = random_split(KFaceDataset, [train_size, total_size - train_size])
        self.train_loader = DataLoader(trainDataset, batch_size=batch_size, num_workers=0, pin_memory=True, shuffle=True)
        self.valid_loader = DataLoader(validDataset, batch_size=batch_size*2, num_workers=0, pin_memory=True, shuffle=False)

        # Call CNN model
        self.model = CNN(class_num).to(self.device)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-8, verbose=False)
        self.early_stopping = EarlyStopping(verbose=True, checkpoint_file=model_file_path)

    def train(self):
        for e in tqdm(range(self.epoch_num)):
            losses, valid_losses = [], []

            # Train
            self.model.train()
            for i, (images, labels) in tqdm(enumerate(self.train_loader), leave=False):
                images, labels = images.to(self.device), labels.to(self.device)
                pred_probs_full = self.model(images)
                loss = self.loss_fn(pred_probs_full, labels)
                _, pred_labels = torch.max(pred_probs_full, dim=-1)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                losses.append(loss.item())
            
            # Validate
            self.model.eval()
            with torch.no_grad():
                for i, (images, labels) in tqdm(enumerate(self.valid_loader), leave=False):
                    images, labels = images.to(self.device), labels.to(self.device)
                    pred_probs_full = self.model(images)
                    loss = self.loss_fn(pred_probs_full, labels)
                    _, pred_labels = torch.max(pred_probs_full, dim=-1)

                    valid_losses.append(loss.item())
            
            self.scheduler.step(loss)
            self.early_stopping(loss, self.model)
            if self.early_stopping.early_stop:
                break

            if e % self.print_interval == 0:
                print(f"EPOCH: {e} AVERAGE TRAINING LOSS: {sum(losses)/len(losses): .2f} AVERAGE VALIDATION LOSS: {sum(valid_losses)/len(valid_losses): .2f}")
