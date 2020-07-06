import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from collections import namedtuple, deque
import random
import time

env = gym.make('LunarLander-v2')
env.seed(0)

from agent import Agent

agent = Agent(env, state_size=8, action_size=4)

def evolution(n_iterations=1000, max_t=2000, gamma=1.0, population=50, std=2.0):
    """Deep Q-Learning.
    
    Params
    ======
        n_iterations (int): number of episodes used to train the agent
        max_t (int): maximum number of timesteps per episode
        gamma (float): discount rate
        population (int): size of population at each iteration
        std (float): standard deviation of additive noise
    """
    scores_deque = deque(maxlen=100)
    scores = []
    current_weights = std*np.random.randn(agent.get_weights_dim())
    previous_reward = 0

    for i_iteration in range(1, n_iterations+1):
        weights_pop = [current_weights + (std*np.random.randn(agent.get_weights_dim())) for i in range(population)]
        rewards = np.array([agent.evaluate(weights, gamma, max_t) for weights in weights_pop])
        #Set min reward to 0 for weight sum
        rewards = rewards - np.full(len(rewards), np.min(rewards))
        sum_rewards = np.sum(rewards)
        rewards = rewards / sum_rewards

        current_weights = np.average(weights_pop, axis=0, weights=rewards)

        reward = agent.evaluate(current_weights, gamma=1.0)
        if reward >= previous_reward:
            std = max(1e-3, std / 2)
        else:
            std = min(2, std * 2)
        previous_reward = reward
        scores_deque.append(reward)
        scores.append(reward)
        
        torch.save(agent.state_dict(), 'checkpoint.pth')
        
        if i_iteration % 4 == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_iteration, np.mean(scores_deque)))

        if np.mean(scores_deque)>=200.0:
            print('\nEnvironment solved in {:d} iterations!\tAverage Score: {:.2f}'.format(i_iteration-100, np.mean(scores_deque)))
            break
    return scores



scores = evolution()


# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig('training_result.png')