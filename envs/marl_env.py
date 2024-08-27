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
from sim_prompters.twod_one_v_one_prompter import twodMARLPrompter


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
            self.prompter = twodMARLPrompter()
        if self.randomize_initial_state and self.dim == 3:
            self.prompter = oneVOnePrompter()
            
        self._obs = None
        self.initial_goal_distance = 0
        self.min_distance = 0
        self.action_dim = len(self.max_ctrl)
        self._np_random = None
        self.adv_max_ctrl = adv_max_ctrl

        self.distance_max = 30

        #handle multiple adversary in prompter and label list
        self.n_adv = 2
        self.agents = ['evader']
        for i in range(n_adv):
            self.agents.append('adversary'+str(i))
        self.prompter.set_num_adv(self.n_adv)

        #track if first goal has been met
        self.goal_count = None

    @property
    def action_space(
            self,
    ) -> gymnasium.Space:
        
        space = {}
        for label in self.agents:
            space[label] = spaces.Box(low=-1, high=1, shape=(self.action_dim,), dtype=np.float32)

        return spaces.Dict(space)

    @property
    def observation_space(
            self,
    ) -> gymnasium.Space:
        
        evader_obs, adversary_obs = self._get_obs()

        space = {}
        for label in self.agents:
            if 'evader' in label:
                space[label] = spaces.Box(low=-np.inf, high=np.inf, shape=(len(evader_obs),), dtype=np.float32)
            if 'adversary' in label:
                space[label] = spaces.Box(low=-np.inf, high=np.inf, shape=(len(adversary_obs),), dtype=np.float32)

        return spaces.Dict(space)

    def reset(self, **kwargs):
        #randomize initial state
        if self.randomize_initial_state:
            prompt = self.prompter.prompt()
            self.sim.set_sat_initial_pos(pos=prompt['sat_pos'])
            adv_poses = []
            for i in range(self.n_adv):
                adv_poses.append(prompt['adv_pos'+str(i)])
            self.sim.set_adversary_initial_pos(poses=adv_poses)
            self.sim.set_sat_goal(goal=prompt['sat_goal'])
            self.initial_goal_distance = np.linalg.norm(prompt['sat_goal'][0:self.dim]-prompt['sat_pos'][0:self.dim])
            self.min_distance = self.initial_goal_distance
        #reset sim, counters, and collect obs
        self._episode += 1
        self._step = 0
        self.goal_count = 0
        self.sim.reset()
        obs = {}
        obs['evader'], obs['adversary'] = self._get_obs()
        return obs, {'episode': self._episode}

    def step(self, action_dict):
        key_map = {}
        
        #preprocess and set model action for adversary
        adversary_control = []
        i = 0
        for key, action in action_dict.items():
            if 'evader' in key:
                self.sim.set_sat_control(self.preprocess_action(action,self.max_ctrl))
                key_map['evader'] = key
            if 'adversary' in key:
                adverasry_control.append(self.preprocess_action(action,self.adv_max_ctrl))
                key_map['adversary'+str(i)] = key
                i += 1 
        self.sim.set_adversary_control(adversary_control)

        obs,rew,terminated,truncated = {},{},{},{}

        #take step
        self.sim.step()
        self._step += 1
        self._train_step += self.parallel_envs

        temp_obs = self._get_obs()
        temp_rew = self._get_rew()

        for label in self.agents:
            obs[key_map[label]] = temp_rew[label]
            rew[key_map[label]] = temp_rew[label]

        #check episode end and adjust reward
        good_term, bad_term, trunc = self._end_episode() #end by collision, end by max episode

        #fill terminated and truncated dict
        for label in self.agents:
            terminated[key_map[label]] = False
            truncated[key_map[label]] = trunc

        if good_term:
            #goal reward/punishment
            if self.goal_count == 0:
                self.goal_count = 1
                good_term = False
                self.sim.set_sat_goal(goal=np.zeros((len(self.sim.get_sat_goal),)))
            for label in self.agents:
                terminated[key_map[label]] = True
                if 'evader' in label:
                    rew[key_map[label]] += 500
                if 'adversary' in label:
                    rew[key_map[label]] -= 1000
        if bad_term:
            #collision punishment
            for label in self.agents:
                terminated[key_map[label]] = True
                if 'evader' in label:
                    rew[key_map[label]] -= 1000
                if 'adversary' in label:
                    rew[key_map[label]] += 500

        return obs, rew, terminated, truncated, {}

    def _end_episode(self) -> bool:
        '''
        each get_end function returns 3 values:
            terminated_bad: episode end to be penalized
            terminated_good: episode end to be rewarded
            truncated: episode end by cutoff
        '''
        collision = self.sim.collision_check()

        return self.get_evader_end(collision=collision)
    
    def _reward(self) -> float:
        norm_dist = self.sim.distance_to_goal() #/60 #normalized distance to goal
        evader_rew, adv_rew = self.get_evader_reward(dist=norm_dist), self.get_adversary_reward(dist=norm_dist)

        rew = {}
        for label in self.agents:
            if 'evader' in label:
                rew[label] = evader_rew
            if 'adversary' in label:
                rew[label] = adv_rew

        return rew
    
    def _get_obs(self) -> OrderedDict:
        obs = super()._get_obs()

        obs = {}
        for label in self.agents:
            if 'evader' in label:
                obs[label] = self.get_evader_obs(obs=obs)
            if 'adversary' in label:
                obs[label] = self.get_adversary_obs(obs=obs,idx=label[-1])

        return obs
    
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
        return rew

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
            idx: int,
    ) -> OrderedDict:
        """Return observation

           rel_evader_state: state of evader relative to advesary
           rel_goal_state: state of the evaders goal relative to the adversary
        """
        adversary_obs = obs['evader_state'] - obs['adversary'+str(idx)+'_state']

        for i in range(self.n_adv):
            if i != idx:
                adv_state = obs['adversary'+str(i)+'_state'] - obs['adversary'+str(idx)+'_state']
                adversary_obs = np.concatenate((adversary_obs,adv_state))

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

        return collision, False, self._step >= self.max_episode_length