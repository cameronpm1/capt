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


class multiAgentWrapper(MultiAgentEnv):
    '''
    wrapper for multiagent envs
    env should handle dict format, wrapper is
    for rllib MultiAgentEnv inheritance
    '''

    def __init__(
            self,
            env: Type[satGymEnv],
    ): 
        super().__init__()
        self.env = env
        self.label = 'evader'

    def get_action_space(
            self,
    ) -> gymnasium.Space:
        return self.env.action_space

    def get_observation_space(
            self,
    ) -> gymnasium.Space:
        return self.env.observation_space
    
    def step(self, action_dict):

        obs,rew,terminated,truncated  = {},{},{},{}
        terminated_all = False
        truncated_all = False

        obs,rew,terminated,truncated,_ = self.env.step(action_dict)

        for key in action_dict.keys():
            if terminated[key]:
                terminated_all = True
            if truncated[key]:
                truncated_all = True

        terminated['__all__'] = terminated_all
        truncated['__all__'] = truncated_all

        return obs,rew,terminated,truncated,{}
    
    def reset(self, **kwargs):
        obs,ep = {},{}
        obs,ep = self.env.reset(**kwargs)
        return obs,ep
    
    def close(self):
        self.env.unwrapped.close()