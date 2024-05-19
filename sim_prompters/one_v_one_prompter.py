import numpy as np
from gymnasium.utils import seeding


class oneVOnePrompter():
    
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
        vec = (self._np_random.random((3,)) - 0.5) * 2
        vec = vec/np.linalg.norm(vec)
        return vec

    def prompt(self):
        sat_range = [0,5] #range of satellite from center
        goal_range = [30,40] #range of goal from satellite 
        adv_range = [15,15] #range of adversary on line between sat and goal
        adv_dev = [0,4] #range of deviation of adversary from straight line between sat and goal

        prompt = {}

        #generate satellite starting position
        vec = self.random_unit_vec()
        sat_pos = vec * (self._np_random.random()*(sat_range[1]-sat_range[0]) + sat_range[0])
        prompt['sat_pos'] = sat_pos

        #generate goal starting position
        vec = self.random_unit_vec()
        goal_rel_sat = vec * (self._np_random.random()*(goal_range[1]-goal_range[0]) + goal_range[0])
        goal_pos = np.zeros((15,))
        goal_pos[0:3] = sat_pos + goal_rel_sat
        prompt['sat_goal'] = goal_pos

        #generate adversary starting position
        vec = self.random_unit_vec()
        goal_dir = goal_rel_sat/np.linalg.norm(goal_rel_sat)
        adv_rel_sat = goal_dir * (self._np_random.random()*(adv_range[1]-adv_range[0]) + adv_range[0])
        vec = self.random_unit_vec()
        dev_vec = vec * (self._np_random.random()*(adv_dev[1]-adv_dev[0]) + adv_dev[0])
        adv_rel_sat_holder = adv_rel_sat.copy()
        adv_rel_sat += dev_vec
        adv_pos = sat_pos + adv_rel_sat
        prompt['adv_pos'] = adv_pos

        return prompt

        

if __name__ == "__main__":
    prompter = oneVOnePrompter()
    prompter.seed()
    prompter.prompt()