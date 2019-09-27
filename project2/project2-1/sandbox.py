#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gym
import numpy as np
import sys
from collections import defaultdict

from mc import *

"""
    This file includes unit test for mc.py
    You could test the correctness of your code by 
    typing 'nosetests -v mc_test.py' in the terminal
"""
env = gym.make('Blackjack-v0')

def test():
    '''mc_control_epsilon_greedy (20 points)'''
    boundaries_key = [(19,10,True),(19,4,True),(18,7,True),(17,9,True),(17,5,True),
             (17,8,False),(17,6,False),(15,6,False),(14,7,False)]
    boundaries_action = [0, 0, 0, 1, 1, 0, 0, 0, 1]
    
    count = 0
    for _ in range(2):
        Q_500k = mc_control_epsilon_greedy(env, n_episodes=1000000, gamma = 1.0, epsilon=0.1)
        policy = dict((k,np.argmax(v)) for k, v in Q_500k.items())
        print([policy[key] for key in boundaries_key])
        if [policy[key] for key in boundaries_key] == boundaries_action:
            count += 1
    
    
    print(len(Q_500k))
    print(count)

test()