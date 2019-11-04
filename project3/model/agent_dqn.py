
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
import os
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
from agent import Agent
from dqn_model import DQN
from collections import namedtuple,deque
import matplotlib.pyplot as plt
import math
"""
you can import any package and define any extra function as you need
"""

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)


class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize everything you need here.
        For example: 
            paramters for neural network  
            initialize Q net and target Q net
            parameters for repaly buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """

        super(Agent_DQN,self).__init__(env)
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.epsilon_start = 1
        self.epsilon_end = 0.02
        self.epsilon_decay = 200000
        self.epsilon = self.epsilon_start
        
        self.gamma = 0.99
        self.env = env
        self.learning_rate = 0.00015
        
        self.buffer_size = 10000
        self.buffer = deque(maxlen=self.buffer_size)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.policy_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(),lr=self.learning_rate)
        self.reward_array = []
        self.reward_x_axis = []
        self.batch_size = 32
        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
            ###########################
            # YOUR IMPLEMENTATION HERE #
            

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.policy_net.load_state_dict(torch.load('policy_model'))
        self.target_net.load_state_dict(self.policy_net.state_dict()) 
        self.policy_net = self.policy_net.to(self.device)
        self.target_net = self.target_net.to(self.device)
        ###########################
        pass
    
    
    def make_action(self, observation, test=False):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        if test:
           self.epsilon = 0
        observation=FloatTensor(observation.reshape((1,84,84,4))).transpose(1,3).transpose(2,3)
        q = self.policy_net(observation).data.cpu().numpy()
        if random.random() > self.epsilon:
            action = np.argmax(q)
        else:
            action = random.randint(0,3)
        return action
    
    def push(self,data):
        """ You can add additional arguments as you need. 
        Push new data to buffer and remove the old one if the buffer is full.
        
        Hints:
        -----
            you can consider deque(maxlen = 10000) list
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.buffer.append(data)
        ###########################
        
        
    def replay_buffer(self,batch_size):
        """ You can add additional arguments as you need.
        Select batch from buffer.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #      
        ###########################
        return random.sample(self.buffer,batch_size)
        
    def play_game(self,start_state):
        action = self.make_action(start_state)
        n_s,r,terminal,_ = self.env.step(action)
        self.push((start_state,action,r,n_s,terminal))
        return n_s,r,terminal
    
    def loss_function(self):
        data = self.replay_buffer(self.batch_size)
        s,a,r,n_s,terminal = zip(*data)
        s = FloatTensor(np.float32(s)).permute(0,3,1,2).to(self.device)
        a = LongTensor(a).to(self.device)
        r = FloatTensor(r).to(self.device)
        n_s = FloatTensor(np.float32(n_s)).permute(0,3,1,2).to(self.device).to(self.device)
        terminal = FloatTensor(terminal).to(self.device)
        q = self.policy_net(s).gather(1,a.unsqueeze(1)).squeeze(1)
        n_q = self.target_net(n_s).detach().max(1)[0]
        expected_q = r + self.gamma * n_q * (1 - terminal)
        loss = F.smooth_l1_loss(q, expected_q.data)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        
    def train(self):
        """
        Implement your training algorithm here
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        rewards_array = []
        reward_ = 0
        best_mean = 0
        print_rate = 100
        last_saved = None
        start_state = self.env.reset()
        for frames in range (3500000):
            #self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-1. *frames / self.epsilon_decay)
            self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * (1-min(frames,self.epsilon_decay)/ self.epsilon_decay)
            n_s,r,terminal = self.play_game(start_state)
            start_state = n_s
            reward_ += r
            if terminal:
                start_state = self.env.reset()
                rewards_array.append(reward_)
                if len(rewards_array) % print_rate==0:
                    print('%%%%%%%%%%%%%%%%%%%%%%%%%')
                    print('Frames = ', frames)
                    print('Current Epsilon = ', self.epsilon)
                    print('Episode = ', len(rewards_array))
                    print('Reward = ', np.mean(rewards_array[-100:]))#sum(rewards_array[-100:]) / 100)
                    print('Buffer Length = ', len(self.buffer))
                    self.reward_array.append(np.mean(rewards_array[-100:]))
                    self.reward_x_axis.append(len(rewards_array))
                    # self.print_graph()
                    if last_saved != None:
                        print("last saved = ", best_mean)
                    print('%%%%%%%%%%%%%%%%%%%%%%%%%')
                reward_ = 0
                
            if len(self.buffer) >= 5000 and frames % 4 ==0:
                    self.loss_function()
                
            if frames % 5000 == 0:
                    print("Target net updated")
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                    
            mean_reward = np.mean(rewards_array[-100:])
            if mean_reward > best_mean and frames % 100==0:
                    print("Saving model with reward = ", mean_reward)
                    best_mean = mean_reward
                    last_saved = mean_reward
                    torch.save(self.policy_net.state_dict(), 'policy_model')
        ###########################
    
    def print_graph(self):
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.plot(self.reward_x_axis,self.reward_array,label='$y = Rewards, $x = episodes')
        ax.legend()
        fig.savefig('plot.png')