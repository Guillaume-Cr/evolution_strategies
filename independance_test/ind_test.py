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
agents = []
for i in range(num_agents):
    env = gym.make('CartPole-v0')
    env.seed(i)
    agent = AgentTest(env, state_size=4, action_size=2, seed=i)
    agents.append(agent)

def evaluate(weights, index):
    print(index, "   ", agents[index].evaluate(weights, gamma=1.0))
    return (index, agents[index].evaluate(weights, gamma=1.0))

def test():
    std=0.1
    np.random.seed(0)
    current_weights = std*np.random.randn(agents[0].get_weights_dim())

    print(current_weights)
    indices = [i for i in range(num_agents)]

    # pool = mp.Pool(num_agents)
    # for j in range(num_agents):
    #     pool.apply(evaluate, args = (current_weights, j,))
    # pool.close()
    # pool.join()

    with concurrent.futures.ThreadPoolExecutor(max_workers=200) as executor:
        futures = list()
        for j in range(num_agents):
            futures.append(executor.submit(evaluate, current_weights, i))

    return

test()