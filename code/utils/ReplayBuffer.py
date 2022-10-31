import random
from tqdm import tqdm
import numpy as np
import torch
from collections import deque

class ReplayBuffer():
    def __init__(self):
        self.buffer = deque()
        self.size_limit = 50000

    def put_data(self, data):
        self.buffer.append(data)
        if len(self.buffer) > self.size_limit:
            self.buffer.popleft()
    
    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        a_lst, r_lst, done_lst = [], [], []
        
        for (i, transition) in tqdm(enumerate(mini_batch), leave=False):
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

        return state, action, reward, next_state, done_mask
    
    def size(self):
        return len(self.buffer)