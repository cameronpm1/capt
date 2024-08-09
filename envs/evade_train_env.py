import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3 import SAC
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
            adversary_model_path: str,
            ctrl_type: str = 'thrust',
            action_scaling_type: str = 'clip',
            randomize_initial_state: bool = False,
            adversary_policy_dir: Optional[str] = None,
            parallel_envs: int = 20,
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

        sim.set_collision_tolerance(tolerance=1) #IMPORTANT (to prevent evade model from learning to be close to adversary)

        if self.randomize_initial_state and self.dim == 2:
            self.prompter = twodOneVOnePrompter()
        if self.randomize_initial_state and self.dim == 3:
            self.prompter = oneVOnePrompter()

        if adversary_policy_dir is None:
            self.controller_policy = Policy.from_checkpoint('/home/cameron/magpie_rl/models/2Dcontrol.pkl')
            self.adversary_controller = lambda pos: self.controller_policy.compute_single_action(pos)
            self.adversary_model = lambda obs: self.adversary_controller(self.heuristic_adversary_policy(obs))
        else:
            self.adversary_policy = Policy.from_checkpoint(evader_policy_dir)
            self.adversary_model = lambda obs: self.adversary_policy.compute_single_action(obs)

        self._obs = None
        self._rew = None
        self.initial_goal_distance = 0
        self.min_distance = 0
        self.action_dim = len(max_ctrl)

    def reset(self, **kwargs):
        if self.randomize_initial_state:
            prompt = self.prompter.prompt()
            self.sim.set_sat_initial_pos(pos=prompt['sat_pos'])
            self.sim.set_adversary_initial_pos(poses=[prompt['adv_pos']])
            self.sim.set_sat_goal(goal=prompt['sat_goal'])
            self.initial_goal_distance = np.linalg.norm(prompt['sat_goal'][0:self.dim]-prompt['sat_pos'][0:self.dim])
            self.min_distance = self.initial_goal_distance
        self._episode += 1
        self._step = 0
        self.sim.reset()
        self._obs = self._get_obs()
        self._rew = 0
        return self._obs, {'episode': self._episode}

    def step(self, action):
        #preprocesses control for sat
        adversary_obs = self.get_adversary_obs()
        adversary_action = self.adversary_model(adversary_obs)
        
        #preprocess and set model action for adversary
        self.sim.set_sat_control(self.preprocess_action(action))
        self.sim.set_adversary_control([self.preprocess_action(adversary_action)])

        #take step
        self.sim.step()
        self._step += 1
        self._train_step += self.parallel_envs
        obs = self._get_obs()
        rew = self._reward()

        #check episode end and adjust reward
        terminated_bad, terminated_good, truncated = self._end_episode() #end by collision, end by max episode

        if terminated_bad:
            rew -= 400
        if terminated_good:
            rew += 400

        return obs, rew, terminated_bad or terminated_good, truncated, {'done': (terminated_bad or terminated_good, truncated), 'reward': rew}
    
    def _end_episode(self) -> bool:

        collision = self.sim.collision_check()
        goal_reached = self.sim.goal_check()

        if self.sim.distance_to_goal() > self.distance_max:
            too_far = True
        else:
            too_far = False

        return collision or too_far, goal_reached, self._step >= self.max_episode_length

    
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

        adversary_obs = OrderedDict()
        #rel evader state
        adversary_obs['rel_evader_state'] = obs['evader_state'] - obs['adversar0_state']
        #rel goal state
        adversary_obs['rel_goal_state'] = obs['goal_state'] - obs['adversary_state']

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

        return target_point


        