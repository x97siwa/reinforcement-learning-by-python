import gym
import gym.envs.toy_text.frozen_lake as fl
from enum import IntEnum
import numpy as np
import random

"""
The implementation depends on this page:
https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py
"""

# 4x4, not slippery
gym.envs.registration.register(
    id='FrozenLakeEasy-v0', 
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'is_slippery': False}
)

# 8x8, not slippery
gym.envs.registration.register(
    id='FrozenLake8x8Easy-v0', 
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'is_slippery': False, 'map_name': '8x8'}
)

_ENVS = {
    ('4x4', True): 'FrozenLake-v0',
    ('4x4', False): 'FrozenLakeEasy-v0',
    ('8x8', True): 'FrozenLake8x8-v0',
    ('8x8', False): 'FrozenLake8x8Easy-v0',
}

class Action(IntEnum):
    LEFT = fl.LEFT
    DOWN = fl.DOWN
    RIGHT = fl.RIGHT
    UP = fl.UP
    
    @classmethod
    def choice(cls):
        return random.choice(list(Action))

class FrozenLake:

    def __init__(self, map_name='4x4', is_slippery=False):
        self.map = fl.MAPS[map_name]
        self.env = gym.make(_ENVS[(map_name, is_slippery)])
        self.action_space = range(self.env.action_space.n)
        self.state_space = range(self.env.observation_space.n)
        
        map_size = len(self.map)
        
        def perform(state, action):
            pos = self._state2pos(state)
            if action==Action.UP:
                pos = (max(0, pos[0]-1), pos[1])
            elif action==Action.DOWN:
                pos = (min(pos[0]+1, map_size-1), pos[1])
            elif action==Action.LEFT:
                pos = (pos[0], max(0, pos[1]-1))
            elif action==Action.RIGHT:
                pos = (pos[0], min(pos[1]+1, map_size-1))
            return self._pos2state(pos)

                
        # transition function
        self.T = np.zeros((len(self.state_space), len(self.action_space), len(self.state_space)))
        for state in self.state_space:
                for action in self.action_space:
                    self.T[state][action][perform(state, action)] = 1/3 if is_slippery else 1
                    if is_slippery:
                        for i in [(action-1)%4, (action+1)%4]:
                            self.T[state][action][perform(state, Action(i))] += 1/3

        # reward function
        self.R = np.zeros((len(self.state_space)))
        self.set_rewards(H=0, F=0, G=1)
    
    
    def set_rewards(self, H=0, F=0, G=1):
        for s in self.state_space:
            row, col = self._state2pos(s)
            if self.map[row][col]=='G':
                self.R[s] = G
            elif self.map[row][col]=='H':
                self.R[s] = H
            else:
                self.R[s] = F

    
    def reset(self):
        return self.env.reset()

    
    def step(self, action):
        state, reward, done, _ = self.env.step(action)
        return state, reward, done
    
    def render(self):
        self.env.render()
    
    
    def close(self):
        self.env.close()
        
    
    def _state2pos(self, s):
        return divmod(s, len(self.map))
    
    
    def _pos2state(self, p):
        return p[0]*len(self.map)+p[1]
