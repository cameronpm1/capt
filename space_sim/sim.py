import sys
#sys.path.insert(1, 'c:/Users/Cameron Mehlman/Documents/Magpie')

import time
import copy
import control
import cvxpy as cp
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from gymnasium.utils import seeding
from typing import Any, Dict, Type, Optional, Union

from envs.gui import gui
from logger import getlogger
from util.util import line_to_point
from dynamics.static_object import staticObject
from dynamics.base_dynamics import baseDynamics
from dynamics.dynamic_object import dynamicObject
from dynamics.sat_dynamics import satelliteDynamics
from dynamics.quad_dynamics import quadcopterDynamics
from trajectory_planning.path_planner import pathPlanner

from ray.rllib.policy.policy import Policy

logger = getlogger(__name__)

class Sim():
    '''
    Single adversary Satellite Simulation

    main_object: our satellite attempting to reach path_planner.goal
    adversary: a list of adversaries in the simulation
    path_planner: path planner for main_object
    control_method: controller for main object used for trajectory tracking
    '''

    def __init__(
            self,
            main_object: Type[dynamicObject],
            path_planner: Type[pathPlanner],
            control_method: str, #MPC,PPO
            kwargs: Optional[Dict[str, Any]],
            point_cloud_radius: float = 5,
            path_point_tolerance: float = 0.01,
            point_cloud_size: float = 3000,
            goal_tolerance: float = 0.001,
            time_tolerance: float = 300,
            collision_tolerance: float = 1,
            perturbation_force: Optional[float] = None,
            plot_cloud: bool = False,
            control_model_path: Optional[str] = 'C:/Users/Cameron Mehlman/Documents/magpie_rl/models/2D control policy', #'C:/Users/Cameron Mehlman/Documents/magpie_rl/models/3DOFcontrol.zip',
            track_point_cloud: bool = True,
    ):
        
        logger.info('Initializing simulation...')
        
        #main variables
        self.main_object = main_object
        self.path_planner = path_planner
        self.point_cloud_radius = point_cloud_radius
        self.point_cloud_size = point_cloud_size
        self.path_point_tolerance = path_point_tolerance
        self.goal_tolerance = goal_tolerance
        self.time_tolerance = time_tolerance
        self.collision_tolerance = collision_tolerance
        self.obstacles = []
        self.dim = self.main_object.dim
        
        #point cloud variables
        self.plot_cloud = plot_cloud
        self.point_cloud_plot = None
        self.point_cloud = None
        self.dynamic_obstacles = False
        self.raw_point_cloud = []
        self.track_point_cloud = track_point_cloud


        if control_method == 'none':
            self.control_method = None
            self.use_controller = False
        elif 'PPO' in control_method:
            kwargs = {
                'modeldir' : control_model_path, 
                'max_ctrl' : kwargs['max_ctrl'],
                'dim' : self.dim,
                }
            self.control_method = getattr(self,control_method)(**kwargs)
            self.use_controller = True
        else:
            kwargs['dynamics'] = main_object.dynamics
            self.control_method = getattr(self,control_method)(**kwargs)
            self.control_method.update_state()
            self.use_controller = True

        #adversary variables
        self.adversary = []
        self.adversary_control_method = []
        self.adversary_path = []
        self.adversary_goal = []
        self.adversary_path_planner = None

        #general EVADE sim variables
        self.time = 0
        self.done = False
        self.reward = 0
        self.current_path = None
        self.expected_time = None
        self.adjustment_count = 0
    
        self.first_goal = True #has first goal been reached?

        self._np_random = None

        logger.info('Simulation initialized')


    def reset(self) -> None:
        logger.info('Resetting simulation')
        self.done = False
        self.first_goal = True
        self.time = 0
        self.reward = 0
        self.adjustment_count = 0
        self.main_object.reset()
        for obstacle in self.obstacles:
            if isinstance(obstacle,dynamicObject):
                obstacle.reset()
        self.update_point_cloud()
        if self.use_controller:
            self.get_new_path()

    def step(
            self,
    ) -> None:
        self.main_object.step()
        if self.use_controller:
            self.control_method.update_state()
        for obstacle in self.obstacles:
            if isinstance(obstacle,dynamicObject):
                obstacle.step()
        if self.dim == 3 and self.dynamic_obstacles and self.track_point_cloud and self.path_planner.distance_to_goal(state=self.main_object.dynamics.get_state()) > 1:
            self.update_point_cloud(propagate=True)
        else:
            self.update_point_cloud(propagate=False)
        self.time += 1
    
    def set_sat_control(
            self,
            control: list[float],
    ) -> None:
        self.main_object.dynamics.set_control(control)

    def set_adversary_control(
            self,
            controls: list[list[float]],
    ) -> None:
        if len(self.adversary) == 0:
            pass
        else:
            for i,control in enumerate(controls):
                self.adversary[i].dynamics.set_control(control)

    def set_sat_initial_pos(
        self,
        pos: list[float],
    ) -> None:
        self.main_object.dynamics.set_initial_pos(pos)

    def set_sat_initial_vel(
        self,
        vel: list[float],
    ) -> None:
        self.main_object.dynamics.set_initial_vel(vel)

    def set_adversary_initial_pos(
        self,
        poses: list[list[float]],
    ) -> None:
        if len(self.adversary) == 0:
            pass
        else:
            for i,pos in enumerate(poses):
                self.adversary[i].dynamics.set_initial_pos(pos)

    def get_sat_pos(self) -> list[float]:
        return self.main_object.dynamics.get_pos()
    
    def get_sat_state(self) -> list[float]:
        return self.main_object.dynamics.get_state()

    def get_adversary_pos(
        self,
        idx: int = 0,
    ) -> list[float]:
        return self.adversary[idx].dynamics.get_pos()

    def get_adversary_vel(
        self,
        idx: int = 0,
    ) -> list[float]:
        return self.adversary[idx].dynamics.get_vel()

    def distance_to_goal(self) -> None:
        return np.linalg.norm(self.path_planner.goal[0:self.dim]-self.main_object.dynamics.get_pos())
    
    def goal_check(self) -> bool:
        if np.linalg.norm(self.path_planner.goal-self.main_object.dynamics.state) < self.goal_tolerance:
            return True
        return False
    
    def adv_goal_proximity(
            self,
            idx: int = 0
    ) -> bool:
        '''
        outputs:
          True - adversary is within collision tollerance of goal
          False - adversary is no within collision tollerance of goal
        '''
        if np.linalg.norm(self.path_planner.goal-self.adversary[idx].dynamics.state) < self.collision_tolerance:
            return True 
        else: 
            return False
    
    def set_sat_goal(
            self,
            goal: list[float],
    ) -> None:
        self.path_planner.set_goal_state(goal)

    def get_sat_goal(
            self,
    ) -> list[float]:

        return self.path_planner.goal

    def distance_to_obstacle(
        self,
        idx: int,
    ) -> float:
        return np.linalg.norm(self.obstacles[idx].dynamics.get_pos()-self.main_object.dynamics.get_pos())
    
    def distance_to_adversary(
        self,
        idx: int,
    ) -> float:
        return np.linalg.norm(self.adversary[idx].dynamics.get_pos()-self.main_object.dynamics.get_pos())

    def min_distance_to_path(
        self,
        pos: list[float],
    ) -> float:
        '''
        path_start = 0.35
        path_end = 0.5
        path_len = len(self.current_path)

        start_idx = int(path_start*path_len)
        end_idx = int(path_start*path_len)
        path = self.current_path[start_idx:end_idx+1]
        '''
        path_point = 0.5
        path_len = len(self.current_path)

        path_idx = int(path_point*path_len)
        path = [self.current_path[path_idx]]

        min_dist = 1000

        for point in path:
            dist = np.linalg.norm(point[0:self.dim]-pos)
            if dist < min_dist:
                min_dist = dist

        return min_dist

    
    def get_object_data(
            self,
            norm: bool = False
    ) -> list[Dict[str, Any]]:
        '''
        package data for GUI
        return:
            main_object points and lines
            current goal point
            point cloud data ('if available')
            final goal point

            returns everything in coordinates realative to the
            final goal so that it is in the center of the plot
        '''
        objects = {}
        if norm:
            objects['final goal'] = np.array(self.path_planner.goal[0:self.dim])
        else:
            objects['final goal'] = np.array(np.zeros((self.dim,)))
        objects['points'] = np.array(copy.deepcopy(self.main_object.temp_mesh['points'])) - objects['final goal']
        if len(self.main_object.temp_mesh['lines']) > 0:
            objects['lines'] = np.array(copy.deepcopy(self.main_object.temp_mesh['lines'])) - objects['final goal']
        else:
            objects['lines'] = copy.deepcopy(self.main_object.temp_mesh['lines'])
        if self.current_path is not None:
            objects['goal'] = np.array(self.current_path)[:,0:2] - objects['final goal']
        else:
            objects['goal'] = self.current_path
        if self.point_cloud_plot is not None and self.plot_cloud:
            objects['point cloud'] = np.array(self.point_cloud_plot) - objects['final goal']
        if not self.plot_cloud and len(self.obstacles) > 0:
            objects['obstacles'] = np.array([obstacle.dynamics.get_pos() for obstacle in self.obstacles]) - objects['final goal']
        if norm:
            objects['final goal']  -= objects['final goal'] 
        else:
            objects['final goal'] = np.array(self.path_planner.goal[0:self.dim])


        return objects
    
    def seed(self, seed):
        #seed = seeding.hash_seed(seed)
        seeds = []
        seeds.append(seed)
        self._np_random, seed = seeding.np_random(seed)
        return seeds

    def set_collision_tolerance(
        self,
        tolerance: float,
    ) -> None:
        self.collision_tolerance = tolerance

    '''
    EVADE SUPPORT FUNCTIONS
    '''

    def compute_evade_control(self): 
        self.check_reached_path_point()
        if self.current_path.size == 0:
            self.first_goal = True
            self.get_new_path()
        main_object_action = self.control_method.compute_action(goal=self.current_path[0],state=self.main_object.dynamics.state)
        return main_object_action

    def compute_main_object_control(
            self,
            goal: list[float],
    ) -> list[float]:
        main_object_action = self.control_method.compute_action(goal=goal,state=self.main_object.dynamics.state)
        return main_object_action

    def compute_schedule(
            self,
            speed: float,
    ) -> None:
        time = []
        time.append(np.linalg.norm(self.current_path[0][0:self.dim]-self.main_object.dynamics.get_pos())/speed)
        for i in range(len(self.current_path)-1):
            time.append(time[-1] + np.linalg.norm(self.current_path[i+1][0:self.dim]-self.current_path[i][0:self.dim])/speed)
        self.expected_time = time

    def check_reached_path_point(self) -> None:
        if np.linalg.norm(self.current_path[0][0:self.dim]-self.main_object.dynamics.get_pos()) < self.path_point_tolerance:
            if self.first_goal:
                self.compute_schedule(self.main_object.dynamics.get_speed())
                self.first_goal = False
                self.get_new_path()
            self.current_path = self.current_path[1:]
            if self.current_path.size > 0:
                self.compute_schedule(self.main_object.dynamics.get_speed())
                point_cloud = self.generate_proximal_point_cloud()
                self.path_planner.update_point_cloud(point_cloud=point_cloud)
                safe = self.path_planner.check_goal_safety(goals=self.current_path,state=self.main_object.dynamics.get_pos())
                if not safe:
                    self.adjustment_count += 1
                    self.current_path = np.array([])

    def get_new_path(self) -> None:
        if self.point_cloud is None:
            point_cloud = self.point_cloud
        else:
            point_cloud = self.generate_proximal_point_cloud()
        new_path = self.path_planner.compute_desired_path(state=self.main_object.dynamics.state, point_cloud=point_cloud)
        self.adjustment_count += 1
        self.current_path = new_path[1:]

    '''
    POINT CLOUD HANDELING FUNCTIONS
    '''    

    def collision_check(self) -> None:
        if self.raw_point_cloud is None:
            pass
        else:
            location = self.main_object.dynamics.get_pos()
            for point in self.raw_point_cloud:
                rel_point = point-location
                if np.linalg.norm(rel_point) < self.collision_tolerance: 
                    return True
        return False               
    
    def generate_proximal_point_cloud(self) -> list[list[float]]:
        '''
        return all points in env within point_cloud_radius 
        returns points location in relation to spacecraft
        '''
        if self.point_cloud is None:
            return None

        location = self.main_object.dynamics.get_pos()

        cloud = []

        for point in self.point_cloud:
                new_point = point-location
                if np.linalg.norm(new_point) < self.point_cloud_radius:
                    cloud.append(new_point)
 
        if len(cloud) == 0:
            return None
        
        return cloud
    
    def update_point_cloud(
            self,
            propagate: bool = False,
        ) -> None:
        self.point_cloud = None
        self.raw_point_cloud = None
        if len(self.obstacles) > 0:
            object_cloud_size = int(self.point_cloud_size/len(self.obstacles))
        for i,obstacle in enumerate(self.obstacles):
            local_point_cloud = np.array(obstacle.point_cloud_from_mesh(n=object_cloud_size))
            if i == 0:
                self.raw_point_cloud = local_point_cloud
            else:
                self.raw_point_cloud = np.concatenate((self.raw_point_cloud,local_point_cloud))
            if not self.first_goal and propagate and isinstance(obstacle,dynamicObject):
                local_point_cloud = self.propagate_point_cloud(obstacle, local_point_cloud)
            if i == 0:
                self.point_cloud = local_point_cloud
            else:
                self.point_cloud = np.concatenate((self.point_cloud,local_point_cloud))

        #dont plot point clouds greater than 2000 points
        #self.point_cloud_plot = copy.deepcopy(self.point_cloud)
        '''
        if self.point_cloud_size > 2000:
            idx = np.random.randint(self.point_cloud_size, size=2000)
            self.point_cloud_plot = self.point_cloud[idx,:]
        else:
            self.point_cloud_plot = self.point_cloud
        '''

    def propagate_point_cloud(
            self,
            obstacle: Type[dynamicObject],
            point_cloud: list[list[float]],
    ) -> list[list[float]]:
        '''
            calculate the point cloud location at each timestep in the future
            for x timesteps and return combined point cloud data. ONLY based
            on obstacles current velocity

            propagate_timesteps: calculate for x timesteps
            obs_vel: velocity of the obstacle
            avg_std: the average of the stds of the obstacles point cloud dist.
            timestep: given obs_vel the amount of time it will take to travel 1 avg_std
        '''

        obs_pos = obstacle.dynamics.get_pos()
        obs_vel = obstacle.dynamics.get_vel()
        obs_speed = obstacle.dynamics.get_speed()
        t_pos = 300

        new_cloud = point_cloud

        for i,goal in enumerate(self.current_path):
            rel_goal = goal[0:self.dim]-obs_pos
            min_distance, dist_traveled = line_to_point(line = obs_vel, point = rel_goal)
            if dist_traveled < 0:
                continue
            else:
                if min_distance < self.path_planner.algorithm.min_distance:
                    time = dist_traveled/obs_speed
                    multiplier = 1/6 #how far to propogate (fraction of total distance between object and impact)
                    if abs(time-self.expected_time[i]) < self.time_tolerance and time < 300: #2 ADVERSARIES
                        #print('Potential obstacle collision detected, propogating obstacle location')
                        std = 0
                        for point in point_cloud:
                            rel_point = point-obs_pos
                            std += (np.dot(rel_point,obs_vel)/np.linalg.norm(obs_vel))**2
                        std /= len(point_cloud)

                        vel_step = obs_vel/obs_speed * std
                        propagations = np.ceil((dist_traveled*(1-multiplier) + obs_speed*t_pos)/std)
                        forward_step = (obs_vel/obs_speed * dist_traveled*multiplier)
                        new_cloud += forward_step
                        for j in range(int(propagations)):
                            prop_obs = (point_cloud - obs_pos)*(j/propagations*3+1) + obs_pos + (j+1)*vel_step + forward_step
                            new_cloud = np.concatenate((new_cloud,prop_obs))
                        return new_cloud
        
        return new_cloud

    def get_voxelized_point_cloud(self) -> None:
        point_cloud = self.generate_proximal_point_cloud()
        self.path_planner.update_point_cloud(point_cloud=point_cloud)
        return self.path_planner.get_voxelized_point_cloud()
    
    '''
    DYNAMIC OBJECT HANDELING FUNCTIONS
    '''
    
    def create_adversary(
            self,
            adversary: Type[dynamicObject],
    ) -> None:
        self.adversary.append(adversary)
        self.add_obstacle(obstacle=adversary)

    def remove_adversary(self,i) -> None:
        adversary = self.adversary.pop(i)
        name = adversary.get_name()
        del adversary
        for i in range(len(self.obstacles)):
            if name == self.obstacles[i].get_name():
                obstacle = self.obstacles.pop(i)
                del obstacle

    def create_adversary_controller(
            self,
            control_model_path: str = '/home/cameron/magpie_rl/models/3DOFcontrol.zip',
    ) -> None:

        self.adversary_control_method.append(getattr(self,'PPOC')(control_model_path))
    
    def compute_adversary_control(
            self,
            goal: list[float],
            idx: int = 0,
    ) -> list[float]:
        adversary_action = self.adversary_control_method[idx].compute_action(goal=goal,state=self.adversary[idx].dynamics.state)
        return adversary_action

    def add_obstacle(
            self,
            obstacle: Union[Type[dynamicObject],Type[staticObject]],
    ) -> None:
        
        obstacle.update_points()
        self.obstacles.append(obstacle)
        if not self.dynamic_obstacles:
            if isinstance(obstacle, dynamicObject):
                self.dynamic_obstacles = True
    
    def remove_obstacle(
            self,
            obstacle_name: str,
    ) -> None:
        
        for i,obstacle in enumerate(self.obstacles):
            if obstacle.name == obstacle_name:
                self.obstacles.pop(i)
        self.update_point_cloud()
        self.get_new_path()

    def get_obstacles_idx(self) -> int:
        obstacle_idx = []
        for i,obstacle in enumerate(self.obstacles):
            if 'obstacle' in obstacle.get_name():
                obstacle_idx.append(i)
        return obstacle_idx
    
    def set_obs_initial_pos(
            self,
            pos: list[float],
            idx: int,
    ) -> None:

        self.obstacles[idx].dynamics.set_initial_pos(pos)
        self.obstacles[idx].update_points()

    class PPOC():

        def __init__(
                self,
                modeldir: str,
                max_ctrl: list[float],
                scale: str = 'clip',
                dim: int = 3,
        ):
            
            logger.info('initializing PPO controller')
        
            self.controller_policy = Policy.from_checkpoint(modeldir)
            self.controller = lambda pos: self.controller_policy.compute_single_action(pos)
            self.scaling_function = getattr(self,'_'+scale)

            self.dim = dim
            self.max_ctrl = max_ctrl[0:self.dim]

            self.state = None
            self.current_goal = None
            

        def update_state(self):
            pass

        def compute_action(
                self,
                goal: list[float],
                state: list[float],
        ) -> list[list[float]]:

            self.state = state.copy()[0:self.dim*2]
            self.current_goal = goal.copy()[0:self.dim*2]
            self.rel_goal = self.current_goal - self.state 

            action,_,_ = self.controller(self.rel_goal)
            scalled_action = self.scaling_function(action)
            if self.dim == 3:
                full_action = np.zeros((9,))
            if self.dim == 2:
                full_action = np.zeros((3,))
            full_action[0:self.dim] = scalled_action[0:self.dim]

            return full_action

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

    
    class MPC():

        def __init__(
                self,
                dynamics: Type[baseDynamics], #only works w/ quadcopterDynamics satelliteDynamics not supported yet
                upper_state_bounds: list[float],
                lower_state_bounds: list[float],
                max_ctrl: Optional[list[float]],
                upper_control_bounds: Optional[list[float]] = None,
                lower_control_bounds: Optional[list[float]] = None,
                horizon: int = 20,
                valued_actions: int = 5,
        ):
            
            logger.info('initializing MPC controller')
        
            self.dynamics = dynamics
            self.horizon = horizon
            self.valued_actions = valued_actions

            self.state_bounds = [np.array(lower_state_bounds),np.array(upper_state_bounds)]
            self.control_bounds = [np.array(lower_control_bounds),np.array(upper_control_bounds)]

            self.initialize_optimization_parameters()
            self.initialize_discrete_matrices()

        def initialize_optimization_parameters(self) -> None:
            self.x = cp.Variable((self.dynamics.state.size, self.horizon+1))
            self.u = cp.Variable((self.dynamics.control.size, self.horizon))
            self.x_init = cp.Parameter(self.dynamics.state.size)

            self.x_init.value = self.dynamics.state

        def initialize_discrete_matrices(self) -> None:
            # Convert continuous time dynamics into discrete time
            sys = control.StateSpace(self.dynamics.A, self.dynamics.B, self.dynamics.C, self.dynamics.D)
            sys_discrete = control.c2d(sys, self.dynamics.timestep*self.dynamics.horizon, method='zoh')

            self.A = np.array(sys_discrete.A)
            self.B = np.array(sys_discrete.B)


        def update_state(self) -> None:
            self.x_init.value = self.dynamics.state

        def compute_action(
                self,
                goal: list[float],
                state: list[float],
        ) -> list[list[float]]:
        
            cost = 0
            constr = [self.x[:, 0] == self.x_init]
            for t in range(self.horizon):
                cost += cp.quad_form(goal - self.x[:, t], self.dynamics.Q) + cp.quad_form(self.u[:, t], self.dynamics.R)
                constr += [self.state_bounds[0] <= self.x[:, t], self.x[:, t] <= self.state_bounds[1]]
                if isinstance(self.dynamics,quadcopterDynamics):
                    constr += [self.x[:, t + 1] == self.A @ self.x[:, t] + self.B @ self.u[:, t]] 
                else:
                    constr += [self.x[:, t + 1] == self.A @ self.x[:, t] + self.B @ self.u[:, t]]

            cost += cp.quad_form(goal-self.x[:, self.horizon], self.dynamics.Q)  # End of trajectory error cost
            problem = cp.Problem(cp.Minimize(cost), constr)

            problem.solve(solver=cp.OSQP, warm_start=True)

            '''
            FOR FUTURE WORK, IF SOLUTION IS NONE 'is None' DOES NOT WORK
            MUST FIX, using if solution.size==1, but this is a bad fix
            solution data type is a numpy.ndarray but when set to None
            python does not recognize this...
            '''

            solution = np.transpose(np.array(self.u[:,0:self.valued_actions].value))

            if solution.size == 1:
                return solution
            elif solution is not None and isinstance(self.dynamics, quadcopterDynamics): #if quadcopter sim
                return np.array([s - np.array([self.dynamics.mass*self.dynamics.g, 0, 0, 0]) for s in solution]).squeeze()
            else:
                return solution.squeeze()
                

'''  

if __name__ == "__main__":

    
    xmin = np.array([-np.inf,  -np.inf,  -np.inf, -np.inf, -np.inf, -np.inf, -0.2, -0.2, -2*np.pi, -.25, -.25, -.25])
    xmax = np.array([np.inf,   np.inf,   np.inf,   np.inf,  np.inf, np.inf, 0.2,  0.2,   2*np.pi,  .25, .25,  .25])

    ymin = np.array([-20,-5,-5,-5])
    ymax = np.array([20,5,5,5])

    kwargs = {
        'upper_state_bounds' : xmax,
        'lower_state_bounds' : xmin,
        'horizon' : 20,
        'valued_actions' : 1,
    }

'''
    

    


