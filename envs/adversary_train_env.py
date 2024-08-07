import time
import numpy as np
from ray.rllib.policy.policy import Policy
from typing import Any, Dict, Type, Optional, Union

from space_sim.sim import Sim
from envs.sat_gym_env import satGymEnv
from dynamics.dynamic_object import dynamicObject
from trajectory_planning.path_planner import pathPlanner
from sim_prompters.one_v_one_prompter import oneVOnePrompter
from sim_prompters.twod_one_v_one_prompter import twodOneVOnePrompter


class adversaryTrainEnv(satGymEnv):

    def __init__(
            self,
            sim: Type[Sim],
            step_duration: float,
            max_episode_length: int,
            max_ctrl: list[float],
            total_train_steps: float,
            evader_policy_dir: str,
            ctrl_type: str = 'thrust',
            action_scaling_type: str = 'clip',
            randomize_initial_state: bool = False,
    ):
        super().__init__(
            sim=sim,
            step_duration=step_duration,
            max_episode_length=max_episode_length,
            max_ctrl=max_ctrl,
            total_train_steps=total_train_steps,
            action_scaling_type=action_scaling_type,
            randomize_initial_state=randomize_initial_state,
        )

        if self.randomize_initial_state and self.dim == 2:
            self.prompter = twodOneVOnePrompter()
        if self.randomize_initial_state and self.dim == 3:
            self.prompter = oneVOnePrompter()

        self.evader_policy = Policy.from_checkpoint(evader_policy_dir)
        self.evader_model = lambda obs: self.policy.compute_single_action(obs)

        self.distance_max = 30
        self.initial_goal_distance = 0


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
        return self._get_obs(), {'episode': self._episode}

    def step(self, action):
        #compute evader control
        evader_obs = self.get_evader_obs
        evader_action = self.evader_model(evader_obs)

        #preprocess and set model action for adversary
        self.sim.set_sat_control(self.preprocess_action(evader_action))
        self.sim.set_adversary_control([self.preprocess_action(action)])

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

        return obs, rew, terminated_bad or terminated_good, truncated, {'done': (terminated_bad or terminated_good, truncated), 'reward': rew}

    
    def _end_episode(self) -> bool:

        collision = self.sim.collision_check()
        goal_reached = self.sim.goal_check()

        if self.sim.distance_to_adversary(idx=0) > self.distance_max:
            too_far = True
        else:
            too_far = False

        return collision or too_far, goal_reached, self._step >= self.max_episode_length
    
    
    def _reward(self) -> float:

        rew = self.sim.distance_to_goal()/self.distance_max

        return rew

    def get_evader_obs(self) -> OrderedDict:
        """Return observation for evader

            obstacle matrix: binary matrix of voxelized state space

            rel_goal_state: goal state of evader in rel coordinates
        """

        obs = OrderedDict()

        # Satellite goal
        evader_state = self.sim.get_sat_state().copy()[0:self.dim*2]
        obs['rel_goal_state'] = np.array(self.sim.get_sat_goal().copy())[0:self.dim*2] - evader_state

        #evade binary point cloud
        obstacle_matrix = self.sim.get_voxelized_point_cloud()
        obs['obstacles_matrix'] = obstacle_matrix

        return obs
