import random
from enum import IntEnum
import numpy as np

class Action(IntEnum):
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3
    
    @classmethod
    def choice(cls):
        return random.choice(list(Action))

class WinterGoldenCave:
    
    def __init__(self, width=4, height=4, hole_num=2, coin_num=2,
                 rand_seed=100, is_slippery=False, slip_prob=0.3334):
        
        self.width = width
        self.height = height
        self.action_space = range(4)
        self.state_space = range(width*height)
        self.current_state = 0
        
        # generate a map
        self.map = []
        _map = ['H']*hole_num + ['C']*coin_num +['F']*(width*height-hole_num-coin_num-2)
        random.seed(rand_seed)
        random.shuffle(_map)
        _map.insert(0, 'S')
        _map.append('G')
        for row in range(height): # convert array to matrix
            self.map.append(_map[row*width:row*width+width])  
    
        def perform(state, action):
            pos = self.state2pos(state)
            row, col = pos
            if action==Action.UP:
                pos = (max(0, row-1), col)
            elif action==Action.DOWN:
                pos = (min(row+1, self.height-1), col)
            elif action==Action.LEFT:
                pos = (row, max(0, col-1))
            elif action==Action.RIGHT:
                pos = (row, min(col+1, self.width-1))
            return self.pos2state(pos)

        # transition function
        self.T = np.zeros((len(self.state_space), len(self.action_space), len(self.state_space)))
        for state in self.state_space:
            for action in self.action_space:
                self.T[state][action][perform(state, action)] = 1-slip_prob if is_slippery else 1
                if is_slippery:
                    for i in [(action-1)%4, (action+1)%4]:
                        self.T[state][action][perform(state, Action(i))] += slip_prob/2
        
        # reward function
        self.R = np.zeros((len(self.state_space)))
        self.set_rewards(F=0, H=0, C=0.5, G=1)


    def set_rewards(self, F=0, H=0, C=0.5, G=1):
        for s in self.state_space:
            row, col = self.state2pos(s)
            if self.map[row][col]=='G':
                self.R[s] = G
            elif self.map[row][col]=='H':
                self.R[s] = H
            elif self.map[row][col]=='C':
                self.R[s] = C
            else:
                self.R[s] = F
    
    
    def step(self, action):
        state = self.current_state
        state = np.random.choice(len(self.state_space), 1, p=self.T[state][action])[0]
        self.current_state = state
        reward = self.R[state]
        row, col = self.state2pos(state)
        done = self.map[row][col] in 'HG'
        return state, reward, done
        
    
    def reset(self):
        self.current_state = 0
        return self.current_state
    

    def state2pos(self, s):
        return divmod(s, self.width)

    
    def pos2state(self, p):
        return p[0]*self.width+p[1]
    

    def __str__(self):
        return '\n'.join(map(lambda row: ''.join(row), self.map))
