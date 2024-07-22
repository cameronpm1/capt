import numpy as np
from gymnasium.utils import seeding
from sim_prompters.twod_control_prompter import twodControlPrompter


class controlPrompter(twodControlPrompter):
    
    def __init__(self):
        super().__init__()

        self.sat_range = [0,0]
        self.vel_range = [0,0.01]
        self.goal_range = [0.1,0.3]

    def seed(self, seed=None):
        seeds = []
        seeds.append(seed)
        self._np_random, seed = seeding.np_random(seed)
        return seeds

    def random_unit_vec(self):
        '''
        returns random unit vector
        '''
        vec = (self._np_random.random((3,)) - 0.5) * 2
        vec = vec/np.linalg.norm(vec)
        return vec

    def prompt(self):
        prompt = {}

        vec = self.random_unit_vec()
        sat_pos = vec * (self._np_random.random()*(self.sat_range[1]-self.sat_range[0]) + self.sat_range[0])
        prompt['sat_pos'] = sat_pos

        vec = self.random_unit_vec()
        sat_vel = vec * (self._np_random.random()*(self.vel_range[1]-self.vel_range[0]) + self.vel_range[0])
        prompt['sat_vel'] = sat_vel

        vec = self.random_unit_vec()
        goal_vec = vec * (self._np_random.random()*(self.goal_range[1]-self.goal_range[0]) + self.goal_range[0])
        goal_pos = np.zeros((15,))
        goal_pos[0:3] = sat_pos + goal_vec
        prompt['sat_goal'] = goal_pos

        return prompt
