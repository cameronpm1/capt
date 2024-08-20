import os
import time
import numpy as np
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


class evadeTrainEnv(satGymEnv):

    def __init__(
            self,
            sim: Type[Sim],
            step_duration: float,
            max_episode_length: int,
            max_ctrl: list[float],
            total_train_steps: float,
            adversary_policy_dir: list[str],
            ctrl_type: str = 'thrust',
            action_scaling_type: str = 'clip',
            randomize_initial_state: bool = False,
            parallel_envs: int = 20,
            adv_max_ctrl: Optional[list[float]] = None,
    ):
        super().__init__(
            sim=sim,
            step_duration=step_duration,
            max_episode_length=max_episode_length,
            max_ctrl=max_ctrl,
            total_train_steps=total_train_steps,
            action_scaling_type=action_scaling_type,
            randomize_initial_state=randomize_initial_state,
            parallel_envs=parallel_envs,
        )

        if self.randomize_initial_state and self.dim == 2:
            self.prompter = twodOneVOnePrompter()
        if self.randomize_initial_state and self.dim == 3:
            self.prompter = oneVOnePrompter()

        adversary_policy_dirs = os.listdir(adversary_policy_dir)

        self.num_policies = 0
        for i,policy in enumerate(adversary_policy_dirs):
            policy_dir = adversary_policy_dir + '/' + policy
            self.num_policies += 1
            policy_label = 'policy' + str(i)
            model_label = 'model' + str(i)
            setattr(self, policy_label, Policy.from_checkpoint(policy_dir))
            setattr(self, model_label, lambda obs: getattr(self, policy_label).compute_single_action(obs))
        self.adversary_model = None
            
        self._obs = None
        self._rew = None
        self.initial_goal_distance = 0
        self.min_distance = 0
        self.action_dim = len(max_ctrl)
        self._np_random = None
        if adv_max_ctrl is None: self.adv_max_ctrl = self.max_ctrl
        else: self.adv_max_ctrl = adv_max_ctrl

        self.distance_max = 60

    def reset(self, **kwargs):
        #pick random adversary model
        model_num = int(self._np_random.integers(0,self.num_policies))
        self.adversary_model = getattr(self,'model'+str(model_num))
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
        self._obs = self._get_obs()
        self._rew = 0
        return self._obs, {'episode': self._episode}

    def step(self, action):
        #preprocesses control for sat
        adversary_obs = self.get_adversary_obs()
        adversary_action,_,_ = self.adversary_model(np.array(adversary_obs))

        #preprocess and set model action for adversary
        self.sim.set_sat_control(self.preprocess_action(action))
        self.sim.set_adversary_control([self.preprocess_action(adversary_action,self.adv_max_ctrl)])

        #take step
        self.sim.step()
        self._step += 1
        self._train_step += self.parallel_envs
        obs = self._get_obs()
        rew = self._reward()

        #check episode end and adjust reward
        terminated_bad, terminated_good, truncated = self._end_episode() #end by collision, end by max episode

        if terminated_bad:
            rew -= (1000-np.clip(self._step,0,1000))
        if terminated_good:
            rew += 1000

        return obs, rew, terminated_bad or terminated_good, truncated, {'done': (terminated_bad or terminated_good, truncated), 'reward': rew}
    
    def _end_episode(self) -> bool:

        collision = self.sim.collision_check()
        goal_reached = self.sim.goal_check()

        if self.sim.distance_to_goal() > self.distance_max:
            too_far = True
        else:
            too_far = False 

        adv_goal_proximity = self.sim.adv_goal_proximity(idx=0)

        return collision or too_far or adv_goal_proximity, goal_reached, self._step >= self.max_episode_length

    
    def _reward(self) -> float:
        dist = self.sim.distance_to_goal()/self.distance_max #inverse of dif between state and goal
        return -1*dist

    def _get_obs(self) -> OrderedDict:
        """Return observation

           only returns evader_state and goal
        """

        obs = OrderedDict()

        # Satellite goal
        evader_state = self.sim.get_sat_state().copy()[0:self.dim*2]
        obs['rel_goal_state'] = np.array(self.sim.get_sat_goal().copy())[0:self.dim*2] - evader_state

        #evade binary point cloud
        obstacle_matrix = self.sim.get_voxelized_point_cloud()
        obs['obstacles_matrix'] = obstacle_matrix

        return obs

    def get_adversary_obs(self) -> OrderedDict:
        """Return observation

           rel_evader_state: state of evader relative to advesary
           rel_goal_state: state of the evaders goal relative to the adversary
        """
        obs = super()._get_obs()

        #adversary_obs = OrderedDict()
        #rel evader state
        adversary_obs = obs['evader_state'] - obs['adversary0_state']
        #rel goal state
        #adversary_obs['rel_goal_state'] = obs['goal_state'] - obs['adversary0_state']

        return adversary_obs
    
    def seed(self, seed=None):  
        seeds = super().seed(seed=seed)
        #initialize random number generator
        self._np_random, seed = seeding.np_random(seed)

        return seeds



        