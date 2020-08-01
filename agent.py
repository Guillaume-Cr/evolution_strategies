import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from collections import deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent(nn.Module):
    def __init__(self, lock, env, state_size, action_size, h_sizes=[256,128], seed=0):
        super(Agent, self).__init__()
        self.lock = lock
        self.env = env
        self.env.reset()
        # state, hidden layer, action sizes
        self.s_size = state_size
        self.h_sizes = h_sizes
        self.a_size = action_size
        self.layer_sizes = [self.s_size]
        self.layer_sizes += self.h_sizes
        self.layer_sizes.append(self.a_size)
        print(self.layer_sizes)

        # define layers
        self.layers = []
        self.layers.append(nn.Linear(self.s_size, self.h_sizes[0]))
        self.layers.append(nn.Linear(self.h_sizes[0], self.h_sizes[1]))
        self.layers.append(nn.Linear(self.h_sizes[1], self.a_size))
        self.seed = torch.manual_seed(seed)

    def forward(self, state):
        x = state
        for i in range(len(self.layers)):
            #print(i)
            x = self.layers[i](x)
            if i != len(self.layers) - 1:
                x = F.relu(x)
        x = F.softmax(x, dim=-1)
        return np.random.choice(self.a_size, p=x.detach().numpy())
        
    def set_weights(self, weights):
        s_size = self.s_size
        h_sizes = self.h_sizes
        a_size = self.a_size
        sizes = self.layer_sizes

        # separate the weights for each layer
        fc_W = [0 for i in range(len(sizes) - 1)]
        fc_b = [0 for i in range(len(sizes) - 1)]
        start = 0
        for i in range(len(sizes) - 1):
            #print("start", start)
            end = start + (sizes[i]*sizes[i+1]) + sizes[i+1]
            #print("end", end)
            fc_W[i] = torch.from_numpy(weights[start : start + sizes[i]*sizes[i+1]].reshape(sizes[i], sizes[i+1]))
            fc_b[i] = torch.from_numpy(weights[start + sizes[i]*sizes[i+1] : end])
            start = end
        
            # set the weights for each layer
            #print(fc_W[i].shape)
            #print(self.layers[i].weight.data.shape)
            self.layers[i].weight.data.copy_(fc_W[i].view_as(self.layers[i].weight.data))
            self.layers[i].bias.data.copy_(fc_b[i].view_as(self.layers[i].bias.data))
    
    def get_weights_dim(self):
        size = 0
        for i in range(len(self.layer_sizes) - 1):
            layer_weights_size = (self.layer_sizes[i] + 1) * (self.layer_sizes[i+1])
            size += layer_weights_size
        return size
        
    def evaluate(self, weights, gamma=1.0, max_t=5000):
        #self.lock.acquire()
        self.set_weights(weights)
        episode_return = 0.0
        state = self.env.reset()
        for t in range(max_t):
            state = torch.from_numpy(state).float().to(device)
            action = self.forward(state)
            
            state, reward, done, _ = self.env.step(action)
            
            episode_return += reward * math.pow(gamma, t)
            if done:
                break
        #self.lock.release()
        return episode_return