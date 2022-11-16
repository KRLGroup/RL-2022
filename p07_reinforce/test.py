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

import pickle



# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)

device = 'cpu'

from policy import Policy
env_id = "CarRacing-v2"
# Create the env
env = gym.make(env_id, continuous=False, domain_randomize=False)

# Create the evaluation env
eval_env = gym.make(env_id, continuous=False, domain_randomize=False, render_mode='human')

MAX_PATIENCE = 100

policy = torch.load('model.pt', map_location=device)
policy = policy.to(device)
policy.eval()

def play_agent(env, policy):
    total_reward = 0
    state = env.reset()
    for i in range(60):
        state,_,_,_,_ = env.step(0)
    step = 0
    done = False
    negative_reward_patience = MAX_PATIENCE
    states = deque(maxlen=4)
    for i in range(policy.n_frames):
        states.append(state)
    while not done:
        action, _ = policy.act(states, exploration=False)
        new_state, reward, done, info,_ = env.step(action)
        states.append(new_state)
        if reward >=0:
            negative_reward_patience = MAX_PATIENCE
        else:
            negative_reward_patience -= 1
            if negative_reward_patience == 0:
                done = True
        if done:
            reward = -100
        total_reward += reward
        env.render()
        if done:
            break
        state = new_state
    print("Total Reward:", total_reward)
    
play_agent(eval_env, policy)