import time
import gymnasium
import numpy as np
from gymnasium import spaces
from collections import OrderedDict
from typing import Any, Dict, Type, Optional, Union
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from space_sim.sim import Sim
from envs.sat_gym_env import satGymEnv
from dynamics.dynamic_object import dynamicObject
from trajectory_planning.path_planner import pathPlanner
from sim_prompters.control_prompter import controlPrompter
from sim_prompters.twod_control_prompter import twodControlPrompter


class multiEnvWrapper(MultiAgentEnv):

    def __init__(
            self,
            envs: list[Type[satGymEnv]],
    ): 
        self.envs = envs
        self.nenvs = len(envs)
        self.labels = ['agent'+str(i) for i in range(self.nenvs)]

    @property
    def action_space(
            self,
    ) -> gymnasium.Space:
        
        space = spaces.Dict()

        for i in range(self.nenvs):
            action_space = self.envs[i].action_space
            space[self.labels[i]] = action_space

        return space

    @property
    def observation_space(
            self,
    ) -> gymnasium.Space:

        space = spaces.Dict()

        for i in range(self.nenvs):
            observation_space = self.envs[i].observation_space
            space[self.labels[i]] = observation_space

        return space
    
    def step(self, action_dict):
        obs,rew,terminated,truncated  = {},{},{},{}
        terminated_all = True
        truncated_all = True

        for key, action in action_dict.items():
            idx = self.labels.index(key)
            obs[key],rew[key],terminated[key],truncated[key],_ = self.envs[idx].step(action)
            if not terminated[key]:
                terminated_all = False
            if not truncated[key]:
                truncated_all = False

        terminated['__all__'] = terminated_all
        truncated['__all__'] = truncated_all

        return obs,rew,terminated,truncated,{}
    
    def reset(self, **kwargs):
        obs,ep = {},{}
        for i in range(self.nenvs):
            obs[self.labels[i]],ep[self.labels[i]] = self.envs[i].reset(**kwargs)
        return obs,ep
    
    def close(self):
        for i in range(self.nenvs):
            self.envs[i].unwrapped.close()