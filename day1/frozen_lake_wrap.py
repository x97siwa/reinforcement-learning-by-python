import gym
import gym.envs.toy_text.frozen_lake as fl
from enum import IntEnum
import numpy as np

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
        self.reward_func = lambda pos: int(self.map[pos[0]][pos[1]]=='G')
        
        def transition_func(pos, action):
            p = np.zeros((len(self.map), len(self.map)))
            p[self._perform(pos, action)] = 1/3 if is_slippery else 1
            if is_slippery:
                for i in [(action.value-1)%4, (action.value+1)%4]:
                    p[self._perform(pos, Action(i))] += 1/3
            return p

        self.transition_func = transition_func

    
    def reset(self):
        return self._state2pos(self.env.reset())

    
    def step(self, action):
        state, reward, done, info = self.env.step(action)
        pos = self._state2pos(state)
        return pos, reward, done, info

    
    def close(self):
        self.env.close()

    
    def _state2pos(self, s):
        return divmod(s, len(self.map))

        
    def _perform(self, pos, action):
        if action==Action.UP:
            return (max(0, pos[0]-1), pos[1])
        elif action==Action.DOWN:
            return (min(pos[0]+1, len(self.map)-1), pos[1])
        elif action==Action.LEFT:
            return (pos[0], max(0, pos[1]-1))
        elif action==Action.RIGHT:
            return (pos[0], min(pos[1]+1, len(self.map)-1))
