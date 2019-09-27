#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gym
import numpy as np
import sys
from collections import defaultdict
from collections import Counter

from td import *
"""
    This file includes unit test for td.py
    You could test the correctness of your code by 
    typing 'nosetests -v td_test.py' in the terminal
"""

env = gym.make('CliffWalking-v0')

def test1():
    '''SARSA (25 points)'''
    test_policy = np.array([[ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  2],
       [ 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0]])
    print(test_policy)
    
    Q_s = sarsa(env, n_episodes = 50000, gamma=1.0, alpha=0.01, epsilon=0.1)
    policy_q = np.array([np.argmax(Q_s[key]) if key in Q_s else -1 for key 
                          in np.arange(48)]).reshape((4,12))
    print(policy_q)
    print(np.allclose(policy_q.shape,(4,12)))
    print(np.allclose(policy_q[2:,],test_policy))

def test2():
    '''Q_learning (25 points)'''
    Q_q = q_learning(env,n_episodes = 10000, gamma=1.0, alpha=0.01, epsilon=0.1)
    policy_q = np.array([np.argmax(Q_q[key]) if key in Q_q else -1 for key 
                         in np.arange(48)]).reshape((4,12))
    test_policy = np.array([[ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  2],
       [ 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0]])
    
    print(policy_q)
    print(np.allclose(policy_q.shape,(4,12)))
    print(np.allclose(policy_q[2:,],test_policy))
    

test1()