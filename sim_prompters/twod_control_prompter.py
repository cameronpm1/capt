import numpy as np
from gymnasium.utils import seeding


class twodControlPrompter():
    
    def __init__(self):
        self._np_random = None

    def seed(self, seed=None):
        seeds = []
        seeds.append(seed)
        self._np_random, seed = seeding.np_random(seed)
        return seeds

    def random_unit_vec(self):
        '''
        returns random unit vector
        '''
        vec = (self._np_random.random((2,)) - 0.5) * 2
        vec = vec/np.linalg.norm(vec)
        return vec

    def prompt(self):
        sat_range = [0,0]
        vel_range = [0,0.01]
        goal_range = [0.1,0.3]

        prompt = {}

        vec = self.random_unit_vec()
        sat_pos = vec * (self._np_random.random()*(sat_range[1]-sat_range[0]) + sat_range[0])
        prompt['sat_pos'] = sat_pos

        vec = self.random_unit_vec()
        sat_vel = vec * (self._np_random.random()*(vel_range[1]-vel_range[0]) + vel_range[0])
        prompt['sat_vel'] = sat_vel

        vec = self.random_unit_vec()
        goal_vec = vec * (self._np_random.random()*(goal_range[1]-goal_range[0]) + goal_range[0])
        goal_pos = np.zeros((6,))
        goal_pos[0:2] = sat_pos + goal_vec
        prompt['sat_goal'] = goal_pos

        return prompt
