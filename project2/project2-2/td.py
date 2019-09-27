#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import random
from collections import defaultdict
#-------------------------------------------------------------------------
'''
    Temporal Difference
    In this problem, you will implememnt an AI player for cliffwalking.
    The main goal of this problem is to get familar with temporal diference algorithm.
    You could test the correctness of your code 
    by typing 'nosetests -v td_test.py' in the terminal.
    
    You don't have to follow the comments to write your code. They are provided
    as hints in case you need. 
'''
#-------------------------------------------------------------------------

def epsilon_greedy(Q, state, nA, epsilon = 0.1):
    """Selects epsilon-greedy action for supplied state.
    
    Parameters:
    -----------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a. 
    state: int
        current state
    nA: int
        Number of actions in the environment
    epsilon: float
        The probability to select a random action, range between 0 and 1
    
    Returns:
    --------
    action: int
        action based current state
     Hints:
        You can use the function from project2-1
    """
    ############################
    # YOUR IMPLEMENTATION HERE #
    # create a uniform profile of possibilities with 1 less action
    p = list(np.ones(nA) * epsilon / (nA))
    # select the greedy action and grant it theextra possibility
    p[np.argmax(Q[state])] += (1.0 - epsilon)
    # randomly select an action according to the profile
    action = random.choices(range(nA), weights = p, k = 1)[0]
    ############################
    return action

def sarsa(env, n_episodes, gamma=1.0, alpha=0.5, epsilon=0.1):
    """On-policy TD control. Find an optimal epsilon-greedy policy.
    
    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor, range between 0 and 1
    alpha: float
        step size, range between 0 and 1
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a. 
    Hints:
    -----
    You could consider decaying epsilon, i.e. epsilon = 0.99*epsilon during each episode.
    """
    
    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    ############################
    # YOUR IMPLEMENTATION HERE #
    # loop each episode
    for trial in range(n_episodes):
        # define decaying epsilon
        epsilon *= 0.99
        # initialize the environment 
        state = env.reset()
        # get an action from policy
        action = epsilon_greedy(Q, state, env.nA, epsilon = epsilon)
        # loop for each step of episode
        isDone = False
        t = 0
        while not isDone:
            t += 1
            # return a reward and new state
            # we use env.step to generate a random card
            result = env.step(action)
            next_state = result[0]
            reward = result[1]
            isDone = result[2]
            # get next action
            next_action = epsilon_greedy(Q, next_state, env.nA, epsilon = epsilon)
            # TD update
            # td_target
            # noticing that we need to use the next step for target
            td_target = reward + gamma * Q[next_state][next_action]
            # td_error
            td_error = td_target - Q[state][action]
            # new Q
            Q[state][action] += alpha * td_error
            # update state
            state = next_state
            # update action
            action = next_action
        print(trial,t)
    ############################
    return Q

def q_learning(env, n_episodes, gamma=1.0, alpha=0.5, epsilon=0.1):
    """Off-policy TD control. Find an optimal epsilon-greedy policy.
    
    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor, range between 0 and 1
    alpha: float
        step size, range between 0 and 1
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a. 
    """
    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    ############################
    # YOUR IMPLEMENTATION HERE #
    # loop each episode
    for trial in range(n_episodes):
        # initialize the environment 
        state = env.reset()
        # loop for each step of episode
        isDone = False
        t = 0
        while not isDone:
            t += 1
            # get an action from policy
            action = epsilon_greedy(Q, state, env.nA, epsilon = epsilon)
            # return a new state, reward and done
            # return a reward and new state
            # we use env.step to generate a random card
            result = env.step(action)
            next_state = result[0]
            reward = result[1]
            isDone = result[2]
            # TD update
            # td_target with best Q
            best_action = np.argmax(Q[next_state])
            td_target = reward + gamma *Q[next_state][best_action]
            # td_error
            td_error = td_target - Q[state][action]
            # new Q
            Q[state][action] += alpha * td_error
            # update state
            state = next_state
        print(trial,t)
    ############################
    return Q
