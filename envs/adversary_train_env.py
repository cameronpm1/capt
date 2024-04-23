import numpy as np
from typing import Any, Dict, Type, Optional, Union


from dynamics.dynamic_object import dynamicObject
from trajectory_planning.path_planner import pathPlanner
from envs.obstacle_avoidance_env import obstacleAvoidanceEnv


class adversaryTrainEnv(satGymEnv):

    def __init__(
            self,
            main_object: Type[dynamicObject],
            path_planner: Type[pathPlanner],
            control_method: str, #MPC
            kwargs: Dict[str, Any],
            point_cloud_radius: float = 5,
            path_point_tolerance: float = 0.1,
            point_cloud_size: float = 3000,
            goal_tolerance: float = 0.0001,
            time_tolerance: float = 300,
            perturbation_force: Optional[float] = None,
            plot_cloud: bool = False,
    ):
        super().__init__(
            main_object=main_object,
            path_planner=path_planner,
            control_method=control_method,
            kwargs=kwargs,
            point_cloud_radius=point_cloud_radius,
            path_point_tolerance=path_point_tolerance,
            point_cloud_size=point_cloud_size,
            goal_tolerance=goal_tolerance,
            time_tolerance=time_tolerance,
            perturbation_force=perturbation_force,
        )


    def step(
            self,
            action: list[float],
    ) -> tuple[list[float], int, bool, list[float]]:
        
        self.collision_check()
        action = self.compute_next_action()
        self.main_object.dynamics.set_control(action)
        self.main_object.step()
        self.control_method.update_state()
        if len(self.adversary) > 0:
            self.adversary_step()
        for obstacle in self.obstacles:
            if isinstance(obstacle,dynamicObject):
                obstacle.step()
        if self.dynamic_obstacles:
            if self.path_planner.distance_to_goal(state=self.main_object.dynamics.state) < 1:
                self.update_point_cloud(propagate=False)
            else:
                self.update_point_cloud(propagate=True) 
        self.check_reached_path_point()
        if self.current_path.size == 0:
            self.first_goal = True
            self.get_new_path()
        self.time += 1
        obs = self.get_observation()
        rew = self.reward()
        done = self.done
        info = action
        return obs, rew, done, info
    
    def get_observation(
            self,
    ) -> list[float]:
        obs = np.concatenate((self.adversary[0].dynamics.get_state(),
                              self.main_object.dynamics.get_state(),
                              self.path_planner.get_goal_state()[0:3]))
        return obs
    
    def get_reward(
            self,
    ):
        pass