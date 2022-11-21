import random
from tqdm import tqdm
import numpy as np
import torch
from collections import deque

class ReplayBuffer():
    def __init__(self, model_type, size_limit):       
        self.buffer = deque()
        self.model_type = model_type
        self.size_limit = size_limit

    def put_data(self, data):
        self.buffer.append(data)
        if len(self.buffer) > self.size_limit:
            self.buffer.popleft()
    
    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)

        if self.model_type == 'PPO':
            a_lst, r_lst, prob_lst, done_lst = [], [], [], []
        elif self.model_type == 'AC' or self.model_type == 'DQN':
            a_lst, r_lst, done_lst = [], [], []
        
        for (i, transition) in tqdm(enumerate(mini_batch), leave=False):
            if self.model_type == 'PPO':
                s, a, r, s_prime, prob, done = transition
            elif self.model_type == 'AC' or self.model_type == 'DQN':
                s, a, r, s_prime, done = transition

            s, s_prime = np.expand_dims(s, 0), np.expand_dims(s_prime, 0) # batch

            if i == 0:
                state = s
                next_state = s_prime
            else:
                state = np.concatenate((state, s), axis=0)
                next_state = np.concatenate((next_state, s_prime), axis=0)

            a_lst.append([a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            r_lst.append([r])
            if self.model_type == 'PPO':
                prob_lst.append([prob])
        
        state, next_state = torch.from_numpy(state), torch.from_numpy(next_state)
        action, done_mask = torch.tensor(a_lst), torch.tensor(done_lst)
        reward = torch.tensor(r_lst)
        if self.model_type == 'PPO':
            prob = torch.tensor(prob_lst)

        if self.model_type == 'PPO':
            return state, action, reward, next_state, prob, done_mask
        elif self.model_type == 'AC' or self.model_type == 'DQN':
            return state, action, reward, next_state, done_mask
    
    def size(self):
        return len(self.buffer)