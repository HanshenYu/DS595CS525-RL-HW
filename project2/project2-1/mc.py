#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import random
from collections import defaultdict
#-------------------------------------------------------------------------
'''
    Monte-Carlo
    In this problem, you will implememnt an AI player for Blackjack.
    The main goal of this problem is to get familar with Monte-Carlo algorithm.
    You could test the correctness of your code 
    by typing 'nosetests -v mc_test.py' in the terminal.
    
    You don't have to follow the comments to write your code. They are provided
    as hints in case you need. 
'''
#-------------------------------------------------------------------------

def initial_policy(observation):
    """A policy that sticks if the player score is >= 20 and his otherwise
    
    Parameters:
    -----------
    observation:
    Returns:
    --------
    action: 0 or 1
        0: STICK
        1: HIT
    """
    ############################
    # YOUR IMPLEMENTATION HERE #
    # get parameters from observation
    score = observation[0]
    # action
    action = score < 20
    ############################
    return action 

def mc_prediction(policy, env, n_episodes, gamma = 1.0):
    """Given policy using sampling to calculate the value function 
        by using Monte Carlo first visit algorithm.
    
    Parameters:
    -----------
    policy: function
        A function that maps an obversation to action probabilities
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor
    Returns:
    --------
    V: defaultdict(float)
        A dictionary that maps from state to value
    """
    # initialize empty dictionaries
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    # a nested dictionary that maps state -> value
    V = defaultdict(float)
    
    ############################
    # YOUR IMPLEMENTATION HERE #
    # loop each episode
    for trial in range(n_episodes):
        # initialize the episode
        state = env.reset()
        states = []
        # generate empty episode list
        episode = []
        # loop until episode generation is done
        isDone = False
        while not isDone:
            # select an action
            action = policy(state)
            # return a reward and new state
            # we use env.step to generate a random card
            result = env.step(action)
            next_state = result[0]
            reward = result[1]
            isDone = result[2]
            # append state, action, reward to episode
            episode += [[state, action, reward],]
            # update state to new state
            state = next_state
        # loop for each step of episode, t = T-1, T-2,...,0
        for index, step in enumerate(episode):
            state, action, reward = step
            # compute G
            G = 0
            # G = sum(reward*gamma**i) from step t
            for i,s in enumerate(episode[index:]):
                G += s[2]* gamma**i 
            # unless state_t appears in states
            if state not in states:
                states += [state]
                # update return_count
                returns_count[state] += 1.
                # update return_sum
                returns_sum[state] += G
                # calculate average return for this state over all sampled episodes
                V[state] = returns_sum[state]/returns_count[state]

    ############################
    
    return V

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
    ------
    With probability (1 − epsilon) choose the greedy action.
    With probability epsilon choose an action at random.
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

def mc_control_epsilon_greedy(env, n_episodes, gamma = 1.0, epsilon = 0.1):
    """Monte Carlo control with exploring starts. 
        Find an optimal epsilon-greedy policy.
    
    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a.
    Hint:
    -----
    You could consider decaying epsilon, i.e. epsilon = epsilon-(0.1/n_episodes) during each episode
    and episode must > 0.    
    """
    
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    ############################
    # YOUR IMPLEMENTATION HERE #
    # loop each episode
    for trial in range(n_episodes):
        # define decaying epsilon
        epsilon -= 0.1/n_episodes
        # initialize the episode
        state = env.reset()
        states = []
        # generate empty episode list
        episode = []
        # loop until episode generation is done
        isDone = False
        while not isDone:
            # get an action from epsilon greedy policy
            action = epsilon_greedy(Q, state, 2, epsilon = epsilon)
            # return a reward and new state
            # we use env.step to generate a random card
            result = env.step(action)
            next_state = result[0]
            reward = result[1]
            isDone = result[2]
            # append state, action, reward to episode
            episode += [[state, action, reward],]
            # update state to new state
            state = next_state
        # loop for each step of episode, t = T-1, T-2,...,0
        for index, step in enumerate(episode):
            state, action, reward = step
            # compute G
            G = 0
            # G = sum(reward*gamma**i) from step t
            for i,s in enumerate(episode[index:]):
                G += s[2]* gamma**i 
            # unless state_t appears in states
            if (state,action) not in states:
                states += [(state,action)]
                # update return_count
                returns_count[(state,action)] += 1.
                # update return_sum
                returns_sum[(state,action)] += G
                # calculate average return for this state over all sampled episodes
                Q[state][action] = returns_sum[(state,action)]/returns_count[(state,action)]
        
    return Q
