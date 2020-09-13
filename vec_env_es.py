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
#Number of agents working in parallel
num_agents = 50
env_fns = [make_env('CartPole-v0', num_agents) for _ in range(num_agents)]
env = AsyncVectorEnv(env_fns)
agent = Agent(env, state_size=4, action_size=2, num_agents=num_agents)

env_test = gym.make('CartPole-v0')
agent_test = AgentTest(env_test, state_size=4, action_size=2)

def sample_weights(current_weight, seed, rng, std):
    rng = np.random.RandomState(seed)
    weights = current_weight + (std*rng.randn(agent.get_weights_dim()))
    print("weights before: ", weights)
    return {seed : weights}

def update_weights(weights, indices, alpha, std):
    reconstructed_weights = np.zeros((len(indices), agent.get_weights_dim()))
    for i in indices:
        rng = np.random.RandomState(i)
        reconstructed_weights[i] = weights + std*rng.randn(agent.get_weights_dim())
    rewards = agent.evaluate(reconstructed_weights, num_agents)
    scaled_rewards = np.zeros(len(rewards))
    for i in indices:
        scaled_rewards[i] = rewards[i]
    scaled_rewards = (scaled_rewards - np.mean(scaled_rewards)) / (np.std(scaled_rewards) + 0.1)
    n = len(rewards)
    deltas = alpha / (n * std) * np.dot(reconstructed_weights.T, scaled_rewards)
    return weights + deltas

def evolution(num_agents, n_iterations=10000, max_t=2000, alpha = 0.01, gamma=1.0, std=0.1):
    """Deep Q-Learning.
    
    Params
    ======
        n_iterations (int): number of episodes used to train the agent
        max_t (int): maximum number of timesteps per episode
        alpha (float): iteration step 
        gamma (float): discount rate
        population (int): size of population at each iteration
        std (float): standard deviation of additive noise
    """
    scores_deque = deque(maxlen=100)
    scores = []
    current_weights = []
    sampled_rewards = {}
    sampled_weights = {}
    previous_reward = 0

    start_time = time.time()
    
    current_weights = std*np.random.randn(agent.get_weights_dim())

    indices = [i for i in range(num_agents)]
    rngs = [np.random.RandomState(i) for i in range(num_agents)]

    for i_iteration in range(1, n_iterations+1):

        seeds = [i for i in range(num_agents)]

        sampled_rewards.clear()
        sampled_weights.clear()
        # with concurrent.futures.ThreadPoolExecutor(max_workers=num_agents) as executor:
        #     futures = list()
        #     for j in range(num_agents):
        #         seed = seeds[j]
        #         rng = rngs[j]
        #         futures.append(executor.submit(sample_weights, current_weights, seed, rng, std))
        #     for future in futures:
        #         return_value = future.result()
        #         sampled_weights.update(return_value)
        
        # for i in range(num_agents):
        #     sampled_rewards.update({i: agent_test.evaluate(sampled_weights[i])})
        
        #sampled_rewards = agent.evaluate(sampled_weights, num_agents, gamma, max_t)
        
        current_weights = update_weights(current_weights, indices, alpha, std)
        
        current_reward = agent_test.evaluate(current_weights)
        scores_deque.append(current_reward)
        scores.append(current_reward)
        
        torch.save(agent.state_dict(), 'checkpoint.pth')

        if i_iteration % 1 == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_iteration, np.mean(scores_deque)))
        if i_iteration % 100 == 0:
            elapsed_time = time.time() - start_time
            print("Duration: ", elapsed_time)

        if np.mean(scores_deque)>=195.0:
            print('\nEnvironment solved in {:d} iterations!\tAverage Score: {:.2f}'.format(i_iteration-100, np.mean(scores_deque)))
            elapsed_time = time.time() - start_time
            print("Training duration: ", elapsed_time)
            break
    return scores


scores = evolution(num_agents)


# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig('training_result.png')