import numpy as np
from gymnasium.utils import seeding
from sim_prompters.twod_control_prompter import twodControlPrompter


class controlPrompter(twodControlPrompter):
    
    def __init__(self):
        super().__init__()

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
        distance_tol = 5
        prompt = {}

        vec = self.random_unit_vec()
        sat_pos = vec * (self._np_random.random()*(self.sat_range[1]-self.sat_range[0]) + self.sat_range[0])
        prompt['sat_pos'] = sat_pos

        vec = self.random_unit_vec()
        sat_vel = vec * (self._np_random.random()*(self.vel_range[1]-self.vel_range[0]) + self.vel_range[0])
        prompt['sat_vel'] = sat_vel

        vec = self.random_unit_vec()
        goal_vec = vec * (self._np_random.random()*(self.goal_range[1]-self.goal_range[0]) + self.goal_range[0])
        goal_pos = np.zeros((12,))
        goal_pos[0:3] = sat_pos + goal_vec
        prompt['sat_goal'] = goal_pos

        if self.n_obs > 1:
            goal_dist = np.linalg.norm(goal_vec)
            center_to_goal = vec*goal_dist/2
            obs_dist = goal_dist/2*1.1
            for i in range(self.n_obs):
                not_safe = True
                while not_safe:
                    vec = self.random_unit_vec()
                    obs_vec = vec * self._np_random.random() * obs_dist
                    obs_pos = center_to_goal + obs_vec
                    if np.linalg.norm(obs_pos-sat_pos) > distance_tol and np.linalg.norm(obs_pos-goal_pos[0:3]) > distance_tol:
                        label = 'obs' + str(i) + '_pos'
                        prompt[label] = obs_pos
                        not_safe = False

        return prompt
