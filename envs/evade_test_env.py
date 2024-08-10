import time
import numpy as np
from gymnasium import spaces
from collections import OrderedDict
from ray.rllib.policy.policy import Policy
from typing import Any, Dict, Type, Optional, Union

from space_sim.sim import Sim
from envs.sat_gym_env import satGymEnv
from dynamics.dynamic_object import dynamicObject
from trajectory_planning.path_planner import pathPlanner
from sim_prompters.one_v_one_prompter import oneVOnePrompter
from sim_prompters.twod_one_v_one_prompter import twodOneVOnePrompter


class evadeTestEnv(satGymEnv):

    def __init__(
            self,
            sim: Type[Sim],
            step_duration: float,
            max_episode_length: int,
            max_ctrl: list[float],
            evader_policy_dir: str,
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
            self.prompter = twodOneVOnePrompter()
        if self.randomize_initial_state and self.dim == 3:
            self.prompter = oneVOnePrompter()

        self.EVADE = False
        if evader_policy_dir is None:
            self.EVADE = True
        else:
            self.evader_policy = Policy.from_checkpoint(evader_policy_dir)
            self.evader_model = lambda obs: self.evader_policy.compute_single_action(obs)

        self.controller_policy = Policy.from_checkpoint('/home/cameron/magpie_rl/models/2D control policy')
        self.adversary_controller = lambda pos: self.controller_policy.compute_single_action(pos)

        self.distance_max = 30
        self.initial_goal_distance = 0
        self.observation_space_flat = None


    def reset(self, **kwargs):
        if self.randomize_initial_state:
            prompt = self.prompter.prompt()
            self.sim.set_sat_initial_pos(pos=prompt['sat_pos'])
            self.sim.set_adversary_initial_pos(poses=[prompt['adv_pos']])
            self.sim.set_sat_goal(goal=prompt['sat_goal'])
            self.initial_goal_distance = np.linalg.norm(prompt['sat_goal'][0:self.dim]-prompt['sat_pos'][0:self.dim])
        self._episode += 1
        self._step = 0
        self.sim.reset()
        self._obs = self._get_obs()
        self._rew = 0
        return self._obs, {'episode': self._episode}

    def step(self, _):
        #compute evader control
        evader_obs = self._get_obs()
        if self.EVADE:
            evader_action = self.sim.compute_evade_control()
        else:
            evader_action = self.evader_model(evader_obs)[0]

        #compute heuristic adversary control
        adversary_obs = self._get_adversary_obs()
        adversary_action = self.heuristic_adversary_policy(obs=adversary_obs)[0]

        #preprocess and set controls
        evader_action = self.preprocess_action(evader_action)
        adversary_action = self.preprocess_action(adversary_action)
        self.sim.set_sat_control(evader_action)
        self.sim.set_adversary_control([adversary_action])

        #take step
        self.sim.step()
        self._step += 1
        obs = self._get_obs()
        rew = self._reward()

        #check episode end and adjust reward
        terminated_bad, terminated_good, truncated = self._end_episode() #end by collision, end by max episode

        if terminated_bad:
            rew -= 400
        if terminated_good:
            rew += 400

        return evader_obs, rew, terminated_bad or terminated_good, truncated, {'done': (terminated_bad or terminated_good, truncated), 'reward': rew}

    
    def _end_episode(self) -> bool:

        collision = self.sim.collision_check()
        goal_reached = self.sim.goal_check()

        return collision, goal_reached, self._step >= self.max_episode_length
    
    
    def _reward(self) -> float:

        rew = self.sim.distance_to_goal()/self.distance_max

        return rew

    def _get_adversary_obs(self) -> OrderedDict:
        """Return observation

           rel_evader_state: state of evader relative to advesary
           rel_goal_state: state of the evaders goal relative to the adversary
        """

        obs = super()._get_obs()

        #rel evader state
        obs['rel_evader_state'] = obs['evader_state'] - obs['adversary0_state']
        #rel goal state
        obs['rel_goal_state'] = obs['goal_state'] - obs['adversary0_state']

        return obs
    
    def _get_obs(self) -> OrderedDict:
        obs = OrderedDict()

        # Satellite goal
        evader_state = self.sim.get_sat_state().copy()[0:self.dim*2]
        obs['rel_goal_state'] = np.array(self.sim.get_sat_goal().copy())[0:self.dim*2] - evader_state

        #evade binary point cloud
        obstacle_matrix = self.sim.get_voxelized_point_cloud()
        obs['obstacles_matrix'] = obstacle_matrix

        return obs

    def heuristic_adversary_policy(
        self,
        obs: dict
    ) -> list[float]:

        evader_pos = obs['rel_evader_state']
        evader_goal = obs['rel_goal_state']

        evader_straight = evader_goal - evader_pos
        block_point = (evader_straight/np.linalg.norm(evader_straight) * 5) + evader_pos
        target_point = block_point/np.linalg.norm(block_point) * 1.5

        return self.adversary_controller(target_point)
