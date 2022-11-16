import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torchvision import transforms
import matplotlib.pyplot as plt
from collections import deque


import numpy as np

class Policy(nn.Module):
    def __init__(self, n_frames, n_actions, hidden_size, img_size=(64,64), device=torch.device('cpu')):
        super(Policy, self).__init__()
        self.hidden_size = hidden_size
        self.n_frames = n_frames
        self.conv1 = nn.Conv2d(n_frames, hidden_size, 7)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, 5)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, n_actions)
        self.gs = transforms.Grayscale()
        self.rs = transforms.Resize(img_size)
        self.device = device

    def preproc_state(self, state):
        # State Preprocessing
        state = state[:83,:].transpose(2,0,1) #Torch wants images in format (channels, height, width)
        state = torch.from_numpy(state)
        state = self.gs(state) # grayscale
        state = self.rs(state) # resize
        return state/255 # normalize

    def forward(self, x):
        # Convolutions
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Global Max Pooling
        batch_size = x.shape[0]
        x = x.reshape(batch_size, self.hidden_size, -1).max(axis=2).values
        
        # Layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x
    
    def act(self, states, exploration=True):
        # Stack 4 states
        state = torch.vstack([self.preproc_state(state) for state in states]).unsqueeze(0).to(self.device)
        
        # Get Action Probabilities
        probs = self.forward(state).cpu()
        
        # Return Action and LogProb
        action = probs.argmax(-1)
        log_prob = None
        if exploration:
            m = Categorical(probs)
            action = m.sample()
            log_prob = m.log_prob(action)
        return action.item(), log_prob

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret