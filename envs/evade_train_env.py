import time
import numpy as np
from stable_baselines3 import PPO
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

        sim.set_collision_tolerance(tolerance=1) #IMPORTANT (to prevent evade model from learning to be close to adversary)

        if self.randomize_initial_state and self.dim == 2:
            self.prompter = twodOneVOnePrompter()
        if self.randomize_initial_state and self.dim == 3:
            self.prompter = oneVOnePrompter()

        self.ctrl_type = ctrl_type
        if 'pos' in self.ctrl_type:
            self.sim.create_adversary_controller()

        self.adversary_model = PPO.load(adversary_model_path)

        self._obs = None
        self._rew = None
        self.initial_goal_distance = 0
        self.min_distance = 0

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
        scalled_action = self.scaling_function(action)
        if 'pos' in self.ctrl_type:
            full_action = self.sim.compute_main_object_control(goal=np.concatenate((scalled_action+self.sim.get_sat_pos(),np.zeros((self.state_dim-3,)))))
        else:
            if self.dim == 3:
                full_action = np.zeros((9,))
                full_action[0:3] = scalled_action
            if self.dim == 2:
                full_action = np.zeros((3,))
                full_action[0:2] = scalled_action
        self.sim.set_sat_control(full_action)
        #preprocess model action for adversary
        adversary_action = self.compute_adversary_control()
        self.sim.set_adversary_control([adversary_action])
        #step
        self.sim.step()
        #record new state
        self._step += 1
        self._train_step += 1
        self._obs = self._get_obs()
        self._rew = self._reward()
        terminated, truncated = self._end_episode() #end by collision, end by max episode
        return self._obs, self._rew, terminated, truncated, {'done': (terminated, truncated), 'reward': self._rew}
    
    def _end_episode(self) -> bool:
        dist = self.sim.distance_to_goal()
        
        terminated, truncated = super()._end_episode()

        return dist >= 1.5*self.initial_goal_distance or terminated, truncated

    
    def _reward(self) -> float:
        dist = self.sim.distance_to_goal()

        if dist < self.min_distance:
            self.min_distance = dist
            rew = 1/dist
        else:
            rew = 0

        #rew = (1/dist) * (1 - self._step/1024) #/self.initial_goal_distance divide by timestep?

        return rew

    def compute_adversary_control(self):
        obs = np.concatenate((self._obs['adversary0_state'],self._obs['evader_state']), axis=None) 

        action, _states = self.adversary_model.predict(obs)
        scalled_action = self.scaling_function(action)

        if 'pos' in self.ctrl_type:
            full_action = self.sim.compute_adversary_control(goal=np.concatenate((scalled_action+self.get_adversary_pos(),np.zeros((self.state_dim-3,)))))
        else:
            full_action = np.zeros((9,))
            full_action[0:3] = scalled_action 

        return full_action
        