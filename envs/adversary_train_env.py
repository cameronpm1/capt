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
            total_train_steps: float,
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

        if self.randomize_initial_state:
            self.prompter = oneVOnePrompter()

        self.ctrl_type = ctrl_type
        if 'pos' in self.ctrl_type:
            self.sim.create_adversary_controller()

        self.initial_goal_distance = 0



    def reset(self, **kwargs):
        if self.randomize_initial_state:
            prompt = self.prompter.prompt()
            self.sim.set_sat_initial_pos(pos=prompt['sat_pos'])
            self.sim.set_adversary_initial_pos(poses=[prompt['adv_pos']])
            self.sim.set_sat_goal(goal=prompt['sat_goal'])
            self.initial_goal_distance = np.linalg.norm(prompt['sat_goal'][0:3]-prompt['sat_pos'][0:3])
        self._episode += 1
        self._step = 0
        self.sim.reset()
        return self._get_obs(), {'episode': self._episode}

    def step(self, action):
        #compute EVADE control for sat
        sat_action = self.sim.compute_evade_control()
        self.sim.set_sat_control(sat_action)
        #preprocess model action for adversary
        scalled_action = self.scaling_function(action)
        if 'pos' in self.ctrl_type:
            full_action = self.sim.compute_adversary_control(goal=np.concatenate((scalled_action+self.get_adversary_pos(),np.zeros((self.state_dim-3,)))))
        else:
            full_action = np.zeros((9,))
            full_action[0:3] = scalled_action
        self.sim.set_adversary_control([full_action])
        #step
        self.sim.step()
        #record new state
        self._step += 1
        self._train_step += 1
        obs = self._get_obs()
        rew = self._reward()
        terminated, truncated = self._end_episode() #end by collision, end by max episode
        return obs, rew, terminated, truncated, {'done': (terminated, truncated), 'reward': rew}

    
    def _end_episode(self) -> bool:
        proximity_tol = 15
        
        terminated, truncated = super()._end_episode()
        dist = self.sim.distance_to_obstacle(idx=0)

        return dist >= proximity_tol or terminated, truncated
    
    
    def _reward(self) -> float:

        '''
        REWARD #1: go to point on path
        '''
        #rew = 1/self.sim.min_distance_to_path(pos=self.sim.get_adversary_pos())
        #if rew > 5: 
        #    rew = 5


        '''
        REWARD #2: distance between satellite and goal
        '''
        #rew = self.sim.distance_to_goal() #add derivative term?
        

        #if np.linalg.norm(self.sim.get_adversary_vel()) > 0.005: 
        #    rew = 0
        

        '''
        CURRICULUM LEARNING (THROUGH REWARD)
        '''

        ts_ratio = self._train_step/self.total_train_steps

        if ts_ratio < 0.4:
            rew = 1/self.sim.min_distance_to_path(pos=self.sim.get_adversary_pos())
            if rew > 5: 
                rew = 5
            rew /= 5
        elif ts_ratio > 0.8:
            rew = self.sim.distance_to_goal()/self.initial_goal_distance
        else:
            rew1 = 1/self.sim.min_distance_to_path(pos=self.sim.get_adversary_pos())
            if rew1 > 5: 
                rew1 = 5
            rew1 /= 5
            rew2 = self.sim.distance_to_goal()
            cr_ratio = (ts_ratio-0.4)/0.4
            rew = (1-cr_ratio)*rew1/5 + cr_ratio*rew2/self.initial_goal_distance

        return rew