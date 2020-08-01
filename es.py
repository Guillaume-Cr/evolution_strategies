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

from agent import Agent

print("Cores", mp.cpu_count())
#Number of agents working in parallel
num_agents = 2
agents = []
lock = threading.Lock()
for i in range(num_agents):
    env = gym.make('CartPole-v0')
    agent = Agent(lock, env, state_size=4, action_size=2, seed=i)
    agents.append(agent)

def sample_reward(current_weight, weights, agent, seed, rng, gamma, max_t, std):
    print("Begin")
    weights = current_weight + (std*rng.randn(agent.get_weights_dim()))
    print("seed: ", seed, ", weights: ", weights)
    print("seed: ", seed, ", weights: ", weights)
    reward = agent.evaluate(weights, gamma, max_t)
    print("seed: ", seed, ", reward: ", reward) 
    #print("End")
    return {seed : reward}

def update_weights(weights, seeds, alpha, std, rewards):
    scaled_perturbations = []
    for i in seeds:
        rng = np.random.RandomState(i)
        scaled_perturbations.append(np.multiply(rewards[i], rng.randn(agents[0].get_weights_dim())))
    scaled_perturbations = (scaled_perturbations - np.mean(scaled_perturbations)) / np.std(scaled_perturbations)
    n = len(scaled_perturbations)
    deltas = alpha / (n * std) * np.sum(scaled_perturbations, axis=0)
    return weights + deltas

def evolution(n_iterations=1000, max_t=2000, alpha = 0.01, gamma=1.0, std=0.1):
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
    times = []
    rewards = {}
    previous_reward = 0

    start_time = time.time()

    np.random.seed(0)
    current_weights = std*np.random.randn(agents[0].get_weights_dim())

    print("current weights", current_weights)

    indexes = [i for i in range(num_agents)]

    for i_iteration in range(1, n_iterations+1):

        seeds = [i for i in range(num_agents)]
        rngs = [np.random.RandomState(i) for i in seeds]
        
        start = time.clock()
        rewards.clear()


        # with concurrent.futures.ThreadPoolExecutor(max_workers=200) as executor:
        #     futures = list()
        #     for j in range(num_agents):
        #         agent, i, seed, rng = agents[j], j, seeds[j], rngs[j]
        #         weights = []
        #         futures.append(executor.submit(sample_reward, current_weights, weights, agent, seed, rng, gamma, max_t, std))
        #     for future in futures:
        #         return_value = future.result()
        #         rewards.update(return_value)
        
        for j in range(num_agents):
            agent, i, seed, rng = agents[j], j, seeds[j], rngs[j]
            weights = []
            rewards.update(sample_reward(current_weights, weights, agent, seed, rng, gamma, max_t, std))

        print(i_iteration, " ", rewards)
        
        current_weights = update_weights(current_weights, seeds, alpha, std, rewards)

        current_reward = agents[0].evaluate(current_weights, gamma=1.0)
        scores_deque.append(current_reward)
        scores.append(current_reward)
        
        end = time.clock()
        times.append(end - start)
        
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
            print("Average sampling time: ", np.mean(times))
            break
    pool.close()
    return scores, times


scores, times = evolution()

print(times)
print(scores)
# plot the scores
# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.plot(np.arange(len(scores)), scores)
# plt.ylabel('Score')
# plt.xlabel('Episode #')
# plt.savefig('training_result.png')

fig2 = plt.figure()
ax = fig2.add_subplot(111)
plt.plot(scores, times)
plt.ylabel('Time')
plt.xlabel('Reward')
plt.savefig('time_reward.png')

