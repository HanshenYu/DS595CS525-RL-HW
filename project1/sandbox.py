from mdp_dp_test import *
import gym
import sys
import numpy as np


def test():
    '''render_single (20 points)'''                 
    print("\n" + "-"*25 + "\nBeginning Policy Iteration\n" + "-"*25)
    random_policy = np.ones([env.nS, env.nA]) / env.nA
    p_pi, V_pi = policy_iteration(env.P, env.nS, env.nA, random_policy,tol=1e-8)
    r_pi = render_single(env, p_pi, False, 50)
    print("total rewards of PI: ",r_pi)
    
    print("\n" + "-"*25 + "\nBeginning Value Iteration\n" + "-"*25)
    V = np.zeros(env.nS)
    p_vi, V_vi = value_iteration(env.P, env.nS, env.nA, V,tol=1e-8)
    r_vi = render_single(env, p_vi, False, 50)
    print("total rewards of VI: ",r_vi)
test()
# print(env.P[0][0][0][3])
