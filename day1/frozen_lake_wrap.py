import gym
import gym.envs.toy_text.frozen_lake as fl
from enum import IntEnum

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

class FrozenLake:

    def __init__(self, map_name='4x4', is_slippery=False):
        self.map = fl.MAPS[map_name]
        self.env = gym.make(_ENVS[(map_name, is_slippery)])
        self.state2pos = {}
        
        nrow = self.env.unwrapped.nrow
        ncol = self.env.unwrapped.ncol
        for i in range(nrow):
            for j in range(ncol):
                self.state2pos[i*nrow+j] = (i, j)


    def reset(self):
        return self.state2pos[self.env.reset()]

    
    def step(self, action):
        state, reward, done, info = self.env.step(action)
        pos = self.state2pos[state]
        return pos, reward, done, info

    
    def close(self):
        self.env.close()
