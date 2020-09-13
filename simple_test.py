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
from gym.vector.tests.utils import make_env, make_slow_env
from gym.vector.async_vector_env import AsyncVectorEnv

import concurrent.futures

from agent import Agent
from agent_test import AgentTest

print("Cores", mp.cpu_count())
if __name__ == '__main__':
    #Number of agents working in parallel
    num_agents = 100
    env_fns = [make_env('CartPole-v0', num_agents) for _ in range(num_agents)]
    env = AsyncVectorEnv(env_fns)
    agent = Agent(env, state_size=4, action_size=2, num_agents=num_agents)

    env_test = gym.make('CartPole-v0')
    agent_test = AgentTest(env_test, state_size=4, action_size=2)

    one_set_of_weights = 0.1*np.random.randn(agent.get_weights_dim())
    all_sets_of_weights = []
    for i in range(num_agents):
        all_sets_of_weights.append(one_set_of_weights)

    start_time = time.time()
    for i in range(100):
        rewards = agent.evaluate(all_sets_of_weights, num_agents)
    print("Time needed for VecEnv approach: ", time.time() - start_time)

    start_time = time.time()
    for i in range(100):
        for i in range(num_agents):
            rewards = agent_test.evaluate(one_set_of_weights)
    print("Time needed for sequential approach: ", time.time() - start_time)
