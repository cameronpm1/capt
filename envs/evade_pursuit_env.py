import time
import gymnasium
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from collections import OrderedDict
from typing import Any, Dict, Type, Optional, Union
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from space_sim.sim import Sim
from envs.sat_gym_env import satGymEnv
from dynamics.dynamic_object import dynamicObject
from trajectory_planning.path_planner import pathPlanner
from sim_prompters.one_v_one_prompter import oneVOnePrompter
from sim_prompters.twod_one_v_one_prompter import twodOneVOnePrompter

from logger import getlogger
logger = getlogger(__name__)

class evadePursuitEnv(MultiAgentEnv):

    def __init__(
            self,
            sim: Type[Sim],
            step_duration: float,
            max_episode_length: int,
            max_ctrl: list[float],
            total_train_steps: float,
            adversary_model_path: str,
            ctrl_type: str = 'thrust',
            action_scaling_type: str = 'clip',
            randomize_initial_state: bool = False,
    ):
        """
        Args:
            sim: Space simulation
            step_duration: Time step length of Sim
            max_episode_length: Max timesteps before Sim terminates
            max_ctrl: Upper bounds on control for sat_dynamics
            normalization_method: 'clip', 'std' or 'scale'
            randomize_initial_state: If True, environment uses
                scenario_prompter to generate random scenarios
            scenario_prompter: initializes random space Sim scenarios
        """
        logger.info("Initializing env ...")

        self._episode = 0
        self._train_step = 0
        self._seed = None
        self.action_dim = len(max_ctrl)
        self.step_duration = step_duration
        self.max_episode_length = max_episode_length
        self.total_train_steps = total_train_steps

        #Initialize Simulation
        self.sim = sim
        self.dim = self.sim.dim

        # Control clipping
        self.max_ctrl = max_ctrl
        self.action_scaling_type = action_scaling_type
        self.scaling_function = getattr(self,'_'+self.action_scaling_type)

        #randomize initial state or not
        self.randomize_initial_state = randomize_initial_state

        self.state_dim = len(self.sim.main_object.dynamics.state)

        sim.set_collision_tolerance(tolerance=1) #IMPORTANT (to prevent evade model from learning to be close to adversary)

        self.distance_cutoff = 60

        if self.randomize_initial_state and self.dim == 2:
            self.prompter = twodOneVOnePrompter()
        if self.randomize_initial_state and self.dim == 3:
            self.prompter = oneVOnePrompter()

        self.agents = {'evader', 'adversary'}
        self.obs_partition_keys = {
            'evader': ['goal','adversary','obstacle'],
            'adversary': ['evader','obstacle']
        }

        self._obs = None
        self._rew = None
        self.initial_goal_distance = 0
        self.min_distance = 0

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
        
        obs = self._get_obs()
        
        space = spaces.Dict({
            'evader': spaces.Box(low=-np.inf, high=np.inf, shape=(len(obs['evader']),), dtype=np.float32),
            'adversary': spaces.Box(low=-np.inf, high=np.inf, shape=(len(obs['adversary']),), dtype=np.float32),
            })

        return space

    def reset(self, **kwargs):
        if self.randomize_initial_state:
            prompt = self.prompter.prompt()
            self.sim.set_sat_initial_pos(pos=prompt['sat_pos'])
            self.sim.set_adversary_initial_pos(poses=[prompt['adv_pos']])
            self.sim.set_sat_goal(goal=prompt['sat_goal'])
            self.initial_goal_distance = np.linalg.norm(prompt['sat_goal'][0:self.dim]-prompt['sat_pos'][0:self.dim])
            self.min_distance = self.initial_goal_distance
        self.sim.reset()
        self._obs = self._get_obs()
        self._episode += 1
        self._step = 0
        self._rew = 0
        return self._obs, {'episode': self._episode}

    def step(self, action_dict):
        #preprocesses control for sat
        #scalled_action = self.scaling_function(action)
        for key, action in action_dict.items():
            scalled_action = self.scaling_function(action)
            if self.dim == 3:
                full_action = np.zeros((9,))
                full_action[0:3] = scalled_action
            if self.dim == 2:
                full_action = np.zeros((3,))
                full_action[0:2] = scalled_action
            if 'evader' in key:
                self.sim.set_sat_control(full_action)
            if 'adversary' in key:
                self.sim.set_adversary_control([full_action])
        #step
        self.sim.step()
        #record new state
        self._step += 1
        self._train_step += 1
        self._obs = self._get_obs()
        self._rew = self._reward()
        terminated_dict, truncated_dict = self._end_episode() #end by collision, end by max episode

        #adjust reward
        for key, terminated in terminated_dict.items():
            if terminated['good']:
                self._rew[key] += 100
            if terminated['bad']:
                self._rew[key] -= 100
            terminated_dict[key] = terminated['good'] or terminated['bad']

        terminated_dict['__all__'] = terminated_dict['evader'] or terminated_dict['adversary']
        truncated_dict['__all__'] = truncated_dict['evader'] or truncated_dict['adversary']

        return self._obs, self._rew, terminated_dict, truncated_dict, {}
    
    def close(self):
        del self.sim
        
    def seed(self, seed=None):
        # Save the seed so we can re-seed during un-pickling
        self._seed = seed

        seeds = [seed]
        seeds.extend(self.sim.seed(seed))
        if self.randomize_initial_state:
            seeds.extend(self.prompter.seed(seed=seed))

        return seeds
    
    def _end_episode(self) -> bool:
        '''
        terminated: ONLY checks for collision between 2 agents or obstacle and evader NOT obstacle and adversary
        truncated: if episode limit has been reached (same for both)
        '''
        collision = self.sim.collision_check()
        goal_reached = self.sim.goal_check()

        dist_to_goal = self.sim.distance_to_goal()
        dist_to_adversary = self.sim.distance_to_adversary(idx=0)

        terminated_dict = {
            'evader': {'bad': dist_to_goal >= self.distance_cutoff or collision, 'good': goal_reached}, 
            'adversary': {'bad': dist_to_adversary >= self.distance_cutoff or collision, 'good': goal_reached}
            }
        
        truncated_dict = {
            'evader': self._step >= self.max_episode_length, 
            'adversary': self._step >= self.max_episode_length
            }

        return terminated_dict, truncated_dict

    
    def _reward(self) -> float:
        dist = np.linalg.norm(self.sim.distance_to_goal())/self.distance_cutoff #normalized distance to goal
        rew1 = dist
        rew2 = dist
        rew1 *= -1
    
        return {'evader': rew1, 'adversary': rew2}
    
    def _get_obs(self) -> OrderedDict:
        obs = OrderedDict()

        # Satellite
        obs['evader_state'] = self.sim.get_sat_state().copy()[0:self.dim*2]

        obs['goal_state'] = np.array(self.sim.get_sat_goal().copy())[0:self.dim*2]

        a = 0
        o = 0
        for obstacle in self.sim.obstacles:
            #Adversaries
            if 'adversary' in obstacle.get_name():
                obs['adversary'+str(a)+'_state'] = obstacle.get_state().copy()[0:self.dim*2]
                a += 1
            #Obstacles
            if 'obstacle' in obstacle.get_name():
                obs['obstacle'+str(o)+'_state'] = obstacle.get_state().copy()[0:self.dim*2]
                o += 1

        evader = np.array([])
        adversary = np.array([])

        for obs_key, val in obs.items():
            for part_key in self.obs_partition_keys['evader']:
                if part_key in obs_key:
                    evader = np.concatenate((evader,val-obs['evader_state']))
            for part_key in self.obs_partition_keys['adversary']:
                if part_key in obs_key:
                    adversary = np.concatenate((adversary,val-obs['adversary0_state']))

        obs.clear()

        obs['evader'] = evader
        obs['adversary'] = adversary

        return obs
    

    '''
    SCALING FUNCTIONS
    '''

    def _clip(
            self,
            action: list[float],
    ) -> list[float]:
        return np.multiply(self.max_ctrl,np.clip(action,a_min=-1,a_max=1))
    
    def _std(
            self,
            action: list[float],
    ) -> list[float]:
        if np.std(action) > 1:
            return np.multiply(self.max_ctrl,action/np.std(action))
        else:
            return np.multiply(self.max_ctrl,action)
    
    def _scale(
            self,
            action: list[float],
    ) -> list[float]:
        if abs(action).max > 1:
            return np.multiply(self.max_ctrl,action/np.linalg.norm(action))
        else:
            return np.multiply(self.max_ctrl,action)