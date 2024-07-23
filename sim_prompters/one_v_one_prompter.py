import numpy as np
from gymnasium.utils import seeding
from twod_one_v_one_prompter import twodOneVOnePrompter


class oneVOnePrompter(twodOneVOnePrompter):
    
    def __init__(self):
        super().__init__()

        self.sat_range = [0,5] #range of satellite from center
        self.goal_range = [30,40] #range of goal from satellite 
        self.adv_range = [8,13] #range of adversary on line between sat and goal
        self.adv_dev = [0,4] #range of deviation of adversary from straight line between sat and goal

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

        #generate satellite starting position
        vec = self.random_unit_vec()
        sat_pos = vec * (self._np_random.random()*(self.sat_range[1]-self.sat_range[0]) + self.sat_range[0])
        prompt['sat_pos'] = sat_pos

        #generate goal and adversary vector position
        vec = self.random_unit_vec()
        goal_rel_sat = vec * (self._np_random.random()*(self.goal_range[1]-self.goal_range[0]) + self.goal_range[0])
        adv_rel_sat = vec * (self._np_random.random()*(self.adv_range[1]-self.adv_range[0]) + self.adv_range[0])

        #compute goal position
        goal_pos = np.zeros((15,))
        goal_pos[0:3] = sat_pos + goal_rel_sat
        prompt['sat_goal'] = goal_pos

        #compute adversary starting position
        vec = self.random_unit_vec()
        dev_vec = vec * (self._np_random.random()*(self.adv_dev[1]-self.adv_dev[0]) + self.adv_dev[0])
        adv_rel_sat += dev_vec
        adv_pos = sat_pos + adv_rel_sat
        prompt['adv_pos'] = adv_pos

        if self.n_obs > 1:
            goal_dist = np.linalg.norm(goal_rel_sat)
            center_to_goal = sat_pos + (goal_rel_sat/np.linalg.norm(goal_rel_sat))*goal_dist/2
            obs_dist = goal_dist/2*0.9
            for i in range(self.n_obs):
                not_safe = True
                while not_safe:
                    vec = self.random_unit_vec()
                    obs_vec = vec * self._np_random.random() * obs_dist
                    obs_pos = center_to_goal + obs_vec
                    if np.linalg.norm(obs_pos-sat_pos) > distance_tol and np.linalg.norm(obs_pos-goal_pos[0:2]) > distance_tol:
                        label = 'obs' + str(i) + '_pos'
                        prompt[label] = obs_pos
                        not_safe = False

        return prompt

        

if __name__ == "__main__":
    prompter = oneVOnePrompter()
    prompter.seed()
    prompter.prompt()