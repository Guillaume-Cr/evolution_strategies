import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from collections import namedtuple, deque
import random
import time
import multiprocessing as mp
import threading

import concurrent.futures

from agent_test import AgentTest

print("Cores", mp.cpu_count())
#Number of agents working in parallel
num_agents = 5
env = gym.make('CartPole-v0')
env.seed(0)
agent = AgentTest(env, state_size=4, action_size=2, seed=0)

def evaluate(weights, index):
    reward = agent.evaluate(weights, gamma=1.0)
    print(index, "   ", reward)
    return (index, reward)

def test():
    std=0.1
    np.random.seed(0)
    current_weights = std*np.random.randn(agent.get_weights_dim())
    print(current_weights)
    indices = [i for i in range(num_agents)]

    for i in range(num_agents):
        evaluate(current_weights, i)
    return

test()