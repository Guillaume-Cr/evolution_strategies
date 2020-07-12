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

env = gym.make('LunarLander-v2')
env.seed(0)

from agent import Agent

#Number of agents working in parallel
num_agents = 1
agents = []
for i in range(num_agents):
    agent = Agent(env, state_size=8, action_size=4, seed=i)
    agents.append(agent)

def sample_reward(current_weight, index, seed, population, gamma, max_t, std):
    np.random.seed(seed)
    print("index ", index)
    weights = [current_weight + (std*np.random.randn(agents[index].get_weights_dim())) for i in range(population)]
    rewards = [agents[index].evaluate(weight, gamma, max_t) for weight in weights]
    return rewards

def update_weights(weights, seeds, alpha, std, rewards):
    scaled_perturbations = []
    i_start = seeds[0]
    for i in seeds:
        np.random.seed(i)
        population = len(rewards[i-i_start])
        for j in range(population):
            scaled_perturbations.append(np.multiply(rewards[i-i_start][j], np.random.randn(agents[i-i_start].get_weights_dim())))
    n = len(scaled_perturbations)
    deltas = alpha / (n * std) * np.sum(scaled_perturbations, axis=0)
    return weights + deltas

def evolution(n_iterations=1000, max_t=2000, alpha = 0.1, gamma=1.0, population=20, std=0.2):
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
    pool = mp.Pool(num_agents)
    scores_deque = deque(maxlen=100)
    scores = []
    current_weights = []
    previous_reward = 0

    start_time = time.time()

    for i in range(num_agents):
        agent = Agent(env, state_size=8, action_size=4, seed=i)
        agents.append(agent)
        current_weights.append(std*np.random.randn(agent.get_weights_dim()))
    
    indexes = [i for i in range(num_agents)]

    for i_iteration in range(1, n_iterations+1):

        seeds = [i+i_iteration for i in range(num_agents)]

        rewards = [pool.apply(sample_reward, args=(weight, i, seed,  population, gamma, max_t, std))
            for agent, weight, i, seed in zip(agents, current_weights, indexes, seeds)]
        
        current_weights = [pool.apply(update_weights, args=(weight, seeds, alpha, std, rewards))
            for agent, weight in zip(agents, current_weights)]

        current_rewards = []
        for i in range(num_agents):
            current_rewards.append(agents[i].evaluate(current_weights[i], gamma=1.0))
        current_reward = np.max(current_rewards)
        scores_deque.append(current_reward)
        scores.append(current_reward)
        
        torch.save(agent.state_dict(), 'checkpoint.pth')
        
        if i_iteration % 4 == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_iteration, np.mean(scores_deque)))
        if i_iteration % 100 == 0:
            elapsed_time = time.time() - start_time
            print("Duration: ", elapsed_time)

        if np.mean(scores_deque)>=200.0:
            print('\nEnvironment solved in {:d} iterations!\tAverage Score: {:.2f}'.format(i_iteration-100, np.mean(scores_deque)))
            elapsed_time = time.time() - start_time
            print("Training duration: ", elapsed_time)
            break
    return scores


scores = evolution()
pool.close()


# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig('training_result.png')