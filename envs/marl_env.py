import os
import time
import gymnasium
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from collections import OrderedDict
from ray.rllib.policy.policy import Policy
from typing import Any, Dict, Type, Optional, Union

from space_sim.sim import Sim
from envs.sat_gym_env import satGymEnv
from dynamics.dynamic_object import dynamicObject
from trajectory_planning.path_planner import pathPlanner
from sim_prompters.one_v_one_prompter import oneVOnePrompter
from sim_prompters.twod_one_v_one_prompter import twodOneVOnePrompter


class MARLEnv(satGymEnv):
    '''
    env for training evader and adversary

    all info is returnrned in tuples, where
    evader is fist, and adversary is second

    ex:   evader_obs, adversary_obs = self._get_obs()
          evader_rew, adverasry_rew = self._get_rew()
    '''

    def __init__(
            self,
            sim: Type[Sim],
            step_duration: float,
            max_episode_length: int,
            sat_max_ctrl: list[float],
            adv_max_ctrl: list[float],
            total_train_steps: float,
            action_scaling_type: str = 'clip',
            randomize_initial_state: bool = False,
            parallel_envs: int = 20,
    ):
        super().__init__(
            sim=sim,
            step_duration=step_duration,
            max_episode_length=max_episode_length,
            max_ctrl=sat_max_ctrl,
            total_train_steps=total_train_steps,
            action_scaling_type=action_scaling_type,
            randomize_initial_state=randomize_initial_state,
            parallel_envs=parallel_envs,
        )

        if self.randomize_initial_state and self.dim == 2:
            self.prompter = twodOneVOnePrompter()
        if self.randomize_initial_state and self.dim == 3:
            self.prompter = oneVOnePrompter()
            
        self._obs = None
        self.initial_goal_distance = 0
        self.min_distance = 0
        self.action_dim = len(self.max_ctrl)
        self._np_random = None
        self.adv_max_ctrl = sat_max_ctrl
        self.adv_max_ctrl = adv_max_ctrl

        self.distance_max = 30

        self.agents = {'evader', 'adversary'}

    @property
    def action_space(
            self,
    ) -> gymnasium.Space:
        
        space = spaces.Dict({
            'evader': spaces.Box(low=-1, high=1, shape=(self.action_dim,), dtype=np.float32),
            'adversary': spaces.Box(low=-1, high=1, shape=(self.action_dim,), dtype=np.float32),
            })

        return space

    @property
    def observation_space(
            self,
    ) -> gymnasium.Space:
        
        evader_obs, adversary_obs = self._get_obs()
        
        space = spaces.Dict({
            'evader': spaces.Box(low=-np.inf, high=np.inf, shape=(len(evader_obs),), dtype=np.float32),
            'adversary': spaces.Box(low=-np.inf, high=np.inf, shape=(len(adversary_obs),), dtype=np.float32),
            })

        return space

    def reset(self, **kwargs):
        #randomize initial state
        if self.randomize_initial_state:
            prompt = self.prompter.prompt()
            self.sim.set_sat_initial_pos(pos=prompt['sat_pos'])
            self.sim.set_adversary_initial_pos(poses=[prompt['adv_pos']])
            self.sim.set_sat_goal(goal=prompt['sat_goal'])
            self.initial_goal_distance = np.linalg.norm(prompt['sat_goal'][0:self.dim]-prompt['sat_pos'][0:self.dim])
            self.min_distance = self.initial_goal_distance
        #reset sim, counters, and collect obs
        self._episode += 1
        self._step = 0
        self.sim.reset()
        obs = {}
        obs['evader'], obs['adversary'] = self._get_obs()
        return obs, {'episode': self._episode}

    def step(self, action_dict):
        key_map = {}
        
        #preprocess and set model action for adversary
        for key, action in action_dict.items():
            if 'evader' in key:
                self.sim.set_sat_control(self.preprocess_action(action,self.max_ctrl))
                key_map['evader'] = key
            if 'adversary' in key:
                self.sim.set_adversary_control([self.preprocess_action(action,self.adv_max_ctrl)])
                key_map['adversary'] = key

        obs,rew,terminated,truncated = {},{},{},{}

        #take step
        self.sim.step()
        self._step += 1
        self._train_step += self.parallel_envs
        obs[key_map['evader']], obs[key_map['adversary']] = self._get_obs()
        rew[key_map['evader']], rew[key_map['adversary']] = self._reward()

        #check episode end and adjust reward
        evader_end, adversary_end = self._end_episode() #end by collision, end by max episode

        if evader_end[0]:
            #collision punishment
            rew[key_map['evader']] -= (1000-np.clip(self._step,0,1000))
        if evader_end[1]:
            #reward for reaching goal
            rew[key_map['evader']] += 1000
        if adversary_end[0]:
            #collision and wandering punishment
            rew[key_map['adversary']] -= 600
        if adversary_end[1]:
            #reward for finding goal before evader
            rew[key_map['adversary']] += 600
            #if evader finds goal, end episode and punish adversary
            rew[key_map['evader']] -= (1000-np.clip(self._step,0,1000))

        terminated[key_map['evader']] = evader_end[0] or evader_end[1]
        terminated[key_map['adversary']] = adversary_end[0] or adversary_end[1]
        truncated[key_map['evader']], truncated[key_map['adversary']] = evader_end[2], adversary_end[2]

        return obs, rew, terminated, truncated, {}

    def _end_episode(self) -> bool:
        '''
        each get_end function returns 3 values:
            terminated_bad: episode end to be penalized
            terminated_good: episode end to be rewarded
            truncated: episode end by cutoff
        '''
        collision = self.sim.collision_check()
        evader_end = self.get_evader_end(collision=collision)
        adversary_end = self.get_adversary_end(collision=collision)

        return evader_end, adversary_end
    
    def _reward(self) -> float:
        norm_dist = self.sim.distance_to_goal()/self.distance_max #normalized distance to goal
        return self.get_evader_reward(dist=norm_dist), self.get_adversary_reward(dist=norm_dist)
    
    def _get_obs(self) -> OrderedDict:
        obs = super()._get_obs()

        return self.get_evader_obs(obs=obs), self.get_adversary_obs(obs=obs)
    
    def get_evader_reward(
            self,
            dist: int
    ) -> float:
        #opposite of normalized distance to goal
        rew = dist 
        return -1*rew
    
    def get_adversary_reward(
            self,
            dist: int
    ) -> float:
        #normalized distance to goal
        rew = dist
        return dist

    def get_evader_obs(
            self,
            obs: OrderedDict,
    ) -> OrderedDict:
        """Return observation

           only returns evader_state and goal
        """

        # Satellite goal
        evader_state = obs['evader_state']
        rel_goal_state = obs['goal_state'] - evader_state

        #evade binary point cloud
        obstacle_matrix = self.sim.get_voxelized_point_cloud()
        obstacles_matrix = obstacle_matrix

        return np.concatenate((rel_goal_state,obstacles_matrix.flatten()))

    def get_adversary_obs(
            self,
            obs: OrderedDict,
    ) -> OrderedDict:
        """Return observation

           rel_evader_state: state of evader relative to advesary
           rel_goal_state: state of the evaders goal relative to the adversary
        """
        adversary_obs = obs['evader_state'] - obs['adversary0_state']

        return adversary_obs
    
    def get_evader_end(
            self,
            collision: bool,
    ) -> bool:
        
        goal_reached = self.sim.goal_check()

        return collision, goal_reached, self._step >= self.max_episode_length
    
    def get_adversary_end(
            self,
            collision: bool,
    ) -> bool:

        goal_reached = self.sim.goal_check()

        if self.sim.distance_to_adversary(idx=0) > self.distance_max:
            too_far = True
        else:
            too_far = False 

        adv_goal_proximity = self.sim.adv_goal_proximity(idx=0)

        return collision or too_far, adv_goal_proximity, self._step >= self.max_episode_length