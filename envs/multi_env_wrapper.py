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
            env: Type[satGymEnv],
            num_agents: Optional[int] = 1,
    ): 
        self.env = env
        self.num_agents = num_agents
        self.label = 'agent0'
    @property
    def action_space(
            self,
    ) -> gymnasium.Space:
        
        space = spaces.Dict()
        action_space = self.env.action_space
        space[self.label] = action_space

        return space

    @property
    def observation_space(
            self,
    ) -> gymnasium.Space:

        space = spaces.Dict()
        observation_space = self.env.observation_space
        space[self.label] = observation_space

        return space
    
    def step(self, action_dict):

        obs,rew,terminated,truncated  = {},{},{},{}
        terminated_all = True
        truncated_all = True

        for key, action in action_dict.items():
            obs[key],rew[key],terminated[key],truncated[key],_ = self.env.step(action)

            if not terminated[key]:
                terminated_all = False
            if not truncated[key]:
                truncated_all = False

        terminated['__all__'] = terminated_all
        truncated['__all__'] = truncated_all

        return obs,rew,terminated,truncated,{}
    
    def reset(self, **kwargs):
        obs,ep = {},{}
        obs[self.label],ep[self.label] = self.env.reset(**kwargs)
        return obs,ep
    
    def close(self):
        self.env.unwrapped.close()