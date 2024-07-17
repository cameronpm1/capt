import numpy as np
from collections import OrderedDict
from typing import Any, Dict, Type, Optional, Union

from space_sim.sim import Sim
from envs.sat_gym_env import satGymEnv
from dynamics.dynamic_object import dynamicObject
from trajectory_planning.path_planner import pathPlanner
from sim_prompters.control_prompter import controlPrompter
from sim_prompters.twod_control_prompter import twodControlPrompter


class controlerTrainEnv(satGymEnv):

    def __init__(
            self,
            sim: Type[Sim],
            step_duration: float,
            max_episode_length: int,
            max_ctrl: list[float],
            action_scaling_type: str = 'clip',
            randomize_initial_state: bool = False,
    ):
        super().__init__(
            sim=sim,
            step_duration=step_duration,
            max_episode_length=max_episode_length,
            max_ctrl=max_ctrl,
            action_scaling_type=action_scaling_type,
            randomize_initial_state=randomize_initial_state,
        )

        if self.randomize_initial_state and self.dim == 2:
            self.prompter = twodControlPrompter()
        if self.randomize_initial_state and self.dim == 3:
            self.prompter = controlPrompter()


    def reset(self, **kwargs):
        if self.randomize_initial_state:
            prompt = self.prompter.prompt()
            self.sim.set_sat_initial_pos(pos=prompt['sat_pos']) #set initial sat position
            self.sim.set_sat_initial_vel(vel=prompt['sat_vel']) #set initial sat velocity
            self.sim.set_sat_goal(goal=prompt['sat_goal']) #set new sat goal
        self._episode += 1
        self._step = 0
        self.sim.reset()
        return self._get_obs(), {'episode': self._episode}

    def step(self, action):
        #scale sat action and set action
        scalled_action = self.scaling_function(action)
        full_action = np.zeros((9,))
        full_action[0:3] = scalled_action
        self.sim.set_sat_control(full_action)
        #take step
        self.sim.step()
        self._step += 1
        obs = self._get_obs()
        rew = self._reward()
        terminated, truncated = self._end_episode() #end by collision, end by max episode

        return obs, rew, terminated, truncated, {'done': (terminated, truncated), 'reward': rew}

    '''
    def seed(self, seed=None):
        # Save the seed so we can re-seed during un-pickling
        self._seed = seed

        # Hash the seed to avoid any correlations
        #seed = seeding.hash_seed(seed)

        # Seed environment components with randomness
        seeds =  super().seed()

        if self.randomize_initial_state:
            print('trying!!!!!!!!!!!!!!!!!!!!!!')
            seeds.extend(self.prompter.seed(seed))

        return seeds
    '''

    def _get_obs(self) -> OrderedDict:
        """Return observation

           only returns sat_state and goal
        """

        obs = OrderedDict()

        # Satellite
        obs['sat_state'] = self.sim.main_object.get_state().copy()

        obs['goal_state'] = np.array(self.sim.get_sat_goal().copy())

        return obs
    
    def _reward(self) -> float:
        dist = np.linalg.norm(self.sim.main_object.get_state()-np.array(self.sim.get_sat_goal())) #inverse of dif between state and goal
        return -1*dist

        

