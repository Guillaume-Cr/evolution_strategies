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

env = gym.make('CartPole-v0')
env.seed(0)

from agent import Agent

print("Cores", mp.cpu_count())
#Number of agents working in parallel
num_agents = 1
agents = []
for i in range(num_agents):
    agent = Agent(env, state_size=4, action_size=2, seed=i)
    agents.append(agent)

def nearest_neighbours(A, k, b):
    A = np.array(A)
    if(len(A) < k):
        k = len(A)
    k_nearest = A[(A).argsort()[:k]]
    distance = []
    for i in range(len(k_nearest)):
        #print("distance ", k_nearest[i])
        distance.append(np.abs(b - k_nearest[i]))
    return np.mean(distance)

def sample_reward(current_weight, index, seed, population, gamma, max_t, std, A, k):
    np.random.seed(seed)
    weights = [current_weight + (std*np.random.randn(agents[index].get_weights_dim())) for i in range(population)]
    rewards = []
    Ns = []
    for weight in weights:
        reward, b = agents[index].evaluate(weight, gamma, max_t)
        #print(b)
        rewards.append(reward)
        Ns.append(nearest_neighbours(A, k, b))
    return {seed : (rewards, Ns)}

def update_weights(weights, seeds, alpha, std, rewards_b, population, a):
    scaled_rewards = []
    scaled_bs = []
    scaled_perturbations = []

    for i in seeds:
        scaled_rewards_i = []
        scaled_bs_i = []
        for j in range(population):
          scaled_rewards_i.append(rewards_b[i][0][j])
          scaled_bs_i.append(rewards_b[i][1][j])
          #print("debug: ", rewards_b[i][1][j])
        scaled_rewards_i = (scaled_rewards_i - np.mean(scaled_rewards_i)) / np.std(scaled_rewards_i)
        if(np.std(scaled_bs_i) != 0):
            scaled_bs_i = (scaled_bs_i - np.mean(scaled_bs_i)) / np.std(scaled_bs_i)
        scaled_rewards.append(scaled_rewards_i)
        scaled_bs.append(scaled_bs_i)

    for i in range(len(scaled_bs)):
        np.random.seed(i)
        #print("scaled_rewards_i: ", len(scaled_rewards[i]))
        for j in range(population):
            #print(scaled_bs[i][j])
            scaled_perturbations.append(np.multiply(scaled_rewards[i][j], np.random.randn(agents[0].get_weights_dim())))
    #scaled_perturbations = (scaled_perturbations - np.mean(scaled_perturbations)) / np.std(scaled_perturbations)
    n = len(scaled_perturbations)
    deltas = alpha / (n * std) * (np.sum(scaled_perturbations, axis=0))
    return weights + deltas

def evolution(n_iterations=1000, max_t=2000, alpha = 0.01, gamma=1.0, population=50, std=0.1, k=10, beta=0.98):
    """Deep Q-Learning.
    
    Params
    ======
        n_iterations (int): number of episodes used to train the agent
        max_t (int): maximum number of timesteps per episode
        alpha (float): iteration step 
        gamma (float): discount rate
        population (int): size of population at each iteration
        std (float): standard deviation of additive noise
        k (int): number of nearest neighbors to take into account
        beta (float): coefficient to reduce the importance of Novelty at each step
    """
    pool = mp.Pool(num_agents)
    scores_deque = deque(maxlen=100)
    scores = []
    current_weights = []
    rewards_b = {}
    previous_reward = 0
    A = []
    a = 1
    start_time = time.time()

    for i in range(num_agents):
        current_weights.append(std*np.random.randn(agents[i].get_weights_dim()))
    
    indexes = [i for i in range(num_agents)]

    #initialize A with first values
    for i in range(num_agents):
        current_reward, b = agents[i].evaluate(current_weights[i], gamma=1.0)
        A.append(b)

    for i_iteration in range(1, n_iterations+1):

        a *= beta

        seeds = [i+i_iteration*num_agents for i in range(num_agents)]

        rewards_b.clear()
        for j in indexes:
            agent, weight, i, seed = agents[j], current_weights[j], j, seeds[j]
            rewards_b.update(pool.apply_async(sample_reward, args=(weight, i, seed, population, gamma, max_t, std, A, k)).get())
        
        current_weights = [pool.apply_async(update_weights, args=(weight, seeds, alpha, std, rewards_b, population, a)).get()
            for agent, weight in zip(agents, current_weights)]

        current_rewards = []
        for i in range(num_agents):
            current_reward, b = agents[i].evaluate(current_weights[i], gamma=1.0)
            current_rewards.append(current_reward)
            #print("b ", b)
            A.append(b)
        current_reward = np.max(current_rewards)
        scores_deque.append(current_reward)
        scores.append(current_reward)
        
        torch.save(agent.state_dict(), 'checkpoint.pth')
        
        if i_iteration % 1 == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_iteration, np.mean(scores_deque)))
        if i_iteration % 100 == 0:
            elapsed_time = time.time() - start_time
            print("Duration: ", elapsed_time)

        if np.mean(scores_deque)>=200.0:
            print('\nEnvironment solved in {:d} iterations!\tAverage Score: {:.2f}'.format(i_iteration-100, np.mean(scores_deque)))
            elapsed_time = time.time() - start_time
            print("Training duration: ", elapsed_time)
            break
    pool.close()
    return scores


scores = evolution()


# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig('training_result.png')