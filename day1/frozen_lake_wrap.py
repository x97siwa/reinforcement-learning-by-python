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
        
        map_size = len(self.map)
        
        def perform(pos, action):
            if action==Action.UP:
                return (max(0, pos[0]-1), pos[1])
            elif action==Action.DOWN:
                return (min(pos[0]+1, map_size-1), pos[1])
            elif action==Action.LEFT:
                return (pos[0], max(0, pos[1]-1))
            elif action==Action.RIGHT:
                return (pos[0], min(pos[1]+1, map_size-1))
        
        # transition function
        # (map_size x map_size) x (action_space.n) x (map_size x map_size)
        self.T = np.zeros((map_size, map_size, self.env.action_space.n, map_size, map_size))
        for row in range(map_size):
            for col in range(map_size):
                for action in range(self.env.action_space.n):
                    pos = (row, col)
                    self.T[pos][action][perform(pos, action)] = 1/3 if is_slippery else 1
                    if is_slippery:
                        for i in [(action-1)%4, (action+1)%4]:
                            self.T[pos][action][perform(pos, Action(i))] += 1/3

        # reward function
        self.R = np.zeros((map_size, map_size))
        for row in range(map_size):
            for col in range(map_size):
                self.R[row][col] = int(self.map[row][col]=='G')

    
    def reset(self):
        return self._state2pos(self.env.reset())

    
    def step(self, action):
        state, reward, done, _ = self.env.step(action)
        pos = self._state2pos(state)
        return pos, reward, done

    
    def render(self):
        self.env.render()
    
    
    def close(self):
        self.env.close()

    
    def _state2pos(self, s):
        return divmod(s, len(self.map))
