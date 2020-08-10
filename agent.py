import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from collections import deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Weights():
    def __init__(self, state_size, action_size, h_sizes):
        self.s_size = state_size
        self.h_sizes = h_sizes
        self.a_size = action_size
        self.layer_sizes = [self.s_size]
        self.layer_sizes += self.h_sizes
        self.layer_sizes.append(self.a_size)
        # define layers
        self.layers = []
        self.layers.append(nn.Linear(self.s_size, self.h_sizes[0]))
        self.layers.append(nn.Linear(self.h_sizes[0], self.h_sizes[1]))
        self.layers.append(nn.Linear(self.h_sizes[1], self.a_size))

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

class Agent(nn.Module):
    def __init__(self, env, state_size, action_size, num_agents, h_sizes=[256,128], seed=0):
        super(Agent, self).__init__()
        self.env = env
        self.weightsVec = [Weights(state_size, action_size, h_sizes) for _ in range(num_agents)]
        self.seed = torch.manual_seed(seed)

    def forward(self, states):
        actions = []
        for i in range(len(states)):
            state = states[i]
            x = state
            for j in range(len(self.weightsVec[i].layers)):
                #print(i)
                x = self.weightsVec[i].layers[j](x)
                if j != len(self.weightsVec[i].layers) - 1:
                    x = F.relu(x)
            x = F.softmax(x, dim=-1)
            actions.append(np.random.choice(self.weightsVec[i].a_size, p=x.detach().numpy()))
        return actions
    
    def get_weights_dim(self):
        return self.weightsVec[0].get_weights_dim()
        
    def evaluate(self, weights, num_agents, gamma=1.0, max_t=5000):
        episode_returns = {}
        terminated = []
        for i in range(num_agents):
            #print("weights during ", weights)
            terminated.append(False)
            self.weightsVec[i].set_weights(weights.get(i))
            episode_returns.update({i: 0})
            
        states = self.env.reset()

        #print("init states: ", states)

        # for t in range(max_t):
        #     states = torch.from_numpy(states).float().to(device)
        #     actions = self.forward(states)
        #     for j in range(num_agents):
        #         if j != 0:
        #             actions[j] = 0 
        #     states, rewards, dones, _ = self.env.step(actions)

        #     #print("Actions: ", actions)
        #     #print("States: ", states)
        #     #print("dones", dones)

        #     for i in range(len(rewards)):
        #         if(terminated[i] == True):
        #             continue
        #         previous_reward = episode_returns[i]
        #         new_reward = rewards[i]
        #         episode_returns.update({i: previous_reward + new_reward * math.pow(gamma, t)})
        #         if dones[i] == True:
        #             terminated[i] = True
            
        #     if False not in terminated:
        #         print ("terminated: ", terminated)
        #         print("episode_returns", episode_returns)
        #         print("T", t)
        #         break
                
        for t in range(max_t):
            states = torch.from_numpy(states).float().to(device)
            actions = self.forward(states)
            #print("actions agent", actions)
            states, rewards, dones, infos = self.env.step(actions)
            #print("Actions: ", actions)
            #print("States: ", states)
             #= self.env.step_wait(timeout=0.1)
            #print("states: ", states)

            for i in range(len(rewards)):
                if(terminated[i] == True):
                    continue
                previous_reward = episode_returns[i]
                new_reward = rewards[i]
                episode_returns.update({i: previous_reward + new_reward * math.pow(gamma, t)})
                if dones[i] == True:
                    terminated[i] = True
                
            #print("dones", dones)

            if False not in terminated:
                #print ("terminated: ", terminated)
                #print("episode_returns", episode_returns)
                #print("T", t)
                break
        #print("episode_returns", episode_returns)
        return episode_returns