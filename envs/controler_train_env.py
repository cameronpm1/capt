import numpy as np
from collections import OrderedDict
from typing import Any, Dict, Type, Optional, Union

from space_sim.sim import Sim
from envs.sat_gym_env import satGymEnv
from dynamics.dynamic_object import dynamicObject
from trajectory_planning.path_planner import pathPlanner
from sim_prompters.control_prompter import controlPrompter
from sim_prompters.twod_control_prompter import twodControlPrompter


class controlerTrainEnv(satGymEnv):

    def __init__(
            self,
            sim: Type[Sim],
            step_duration: float,
            max_episode_length: int,
            max_ctrl: list[float],
            total_train_steps: float,
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
            self.prompter = twodControlPrompter()
        if self.randomize_initial_state and self.dim == 3:
            self.prompter = controlPrompter()

        self.obs_idx = self.sim.get_obstacles_idx()
        self.n_obs = len(self.obs_idx)

        if self.n_obs > 0:
            self.prompter.set_num_obstacles(self.n_obs)

        self.distance_max = 60


    def reset(self, **kwargs):
        if self.randomize_initial_state:
            prompt = self.prompter.prompt()
            self.sim.set_sat_initial_pos(pos=prompt['sat_pos']) #set initial sat position
            #self.sim.set_sat_initial_vel(vel=prompt['sat_vel']) #set initial sat velocity
            self.sim.set_sat_goal(goal=prompt['sat_goal']) #set new sat goal
            for i in range(self.n_obs):
                label = 'obs' + str(i) + '_pos'
                self.sim.set_obs_initial_pos(pos=prompt[label],idx=self.obs_idx[i])
        self._episode += 1
        self._step = 0
        self.sim.reset()
        return self._get_obs(), {'episode': self._episode}

    def step(self, action):
        '''
        TO DO:
            integrate get_voxelized_point_cloud() so that RL observation is evade state representation
        '''
        #scale sat action and set action
        scalled_action = self.scaling_function(action)
        if self.dim == 3:
            full_action = np.zeros((9,))
        if self.dim ==2:
            full_action = np.zeros((3,))
        full_action[0:self.dim] = scalled_action
        self.sim.set_sat_control(full_action)
        #take step
        self.sim.step()
        self._step += 1
        obs = self._get_obs()
        rew = self._reward()
        terminated_bad, terminated_good, truncated = self._end_episode() #end by collision, end by max episode

        if terminated_bad:
            rew -= 600

        return obs, rew, terminated_bad or terminated_good, truncated, {'done': (terminated_bad or terminated_good, truncated), 'reward': rew}

    def _get_obs(self) -> OrderedDict:
        """Return observation

           only returns evader_state and goal
        """

        obs = super()._get_obs()

        # Satellite
        obs['rel_goal_state'] = obs['goal_state'] - obs['evader_state']

        for i in range(self.n_obs):
            label = 'rel_obstacle'+str(i)+'_state'
            obs[label] = obs['obstacle'+str(i)+'_state'] - obs['evader_state']

        return obs
    
    def _reward(self) -> float:
        dist = np.linalg.norm(self.sim.main_object.get_state()-np.array(self.sim.get_sat_goal()))/self.distance_max #inverse of dif between state and goal
        return -1*dist
    
    def _end_episode(self) -> bool:

        collision = self.sim.collision_check()
        goal_reached = self.sim.goal_check()

        if self.sim.distance_to_goal() > self.distance_max:
            too_far = True
        else:
            too_far = False

        return collision or too_far, goal_reached, self._step >= self.max_episode_length

        

