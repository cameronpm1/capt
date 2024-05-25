import time
import numpy as np
from typing import Any, Dict, Type, Optional, Union

from space_sim.sim import Sim
from envs.sat_gym_env import satGymEnv
from dynamics.dynamic_object import dynamicObject
from trajectory_planning.path_planner import pathPlanner
from sim_prompters.one_v_one_prompter import oneVOnePrompter


class adversaryTrainEnv(satGymEnv):

    def __init__(
            self,
            sim: Type[Sim],
            step_duration: float,
            max_episode_length: int,
            max_ctrl: list[float],
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

        if self.randomize_initial_state:
            self.prompter = oneVOnePrompter()

    def reset(self, **kwargs):
        if self.randomize_initial_state:
            prompt = self.prompter.prompt()
            self.sim.set_sat_initial_pos(pos=prompt['sat_pos'])
            self.sim.set_adversary_initial_pos(poses=[prompt['adv_pos']])
            self.sim.set_sat_goal(goal=prompt['sat_goal'])
        self._episode += 1
        self._step = 0
        self.sim.reset()
        return self._get_obs(), {'episode': self._episode}

    def step(self, action):

        sat_action = self.sim.compute_evade_control()
        self.sim.set_sat_control(sat_action)

        scalled_action = self.scaling_function(action)
        full_action = np.zeros((9,))
        full_action[0:3] = scalled_action
        self.sim.set_adversary_control([full_action])
        self.sim.step()

        self._step += 1
        obs = self._get_obs()
        rew = self._reward()
        terminated, truncated = self._end_episode() #end by collision, end by max episode
        
        return obs, rew, terminated, truncated, {'done': (terminated, truncated), 'reward': rew}

    def _end_episode(self) -> bool:
        proximity_tol = 15

        terminated, truncated = super()._end_episode()
        dist = self.sim.distance_to_obstacle(idx=0)
        print(dist)

        return terminated, truncated
    
    def _reward(self) -> float:
        '''
        sat_pos = self.sim.main_object.dynamics.get_pos()
        sat_vel = self.sim.main_object.dynamics.get_vel()
        goal_pos = self.sim.get_sat_goal()[0:3]

        sat_to_goal = goal_pos - sat_pos
        proj = np.dot(sat_to_goal,sat_vel)/(np.linalg.norm(sat_to_goal)**2)*sat_to_goal
        proj_mag = np.linalg.norm(proj)

        angle = np.arccos(np.dot(sat_to_goal,sat_vel)/(np.linalg.norm(sat_to_goal)*np.linalg.norm(sat_vel)))
        if abs(angle) < np.pi/2:
            mult = -1
        else:
            mult = 1

        return proj_mag*mult 
        '''
        return self.sim.get_distance_to_goal()