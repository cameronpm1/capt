import os
import sys
import time
import hydra
import gymnasium
import numpy as np
import pyvista as pv
from gymnasium import spaces
from astropy import units as u
from omegaconf import DictConfig, OmegaConf
from gymnasium.wrappers import FilterObservation, FlattenObservation

sys.path.insert(1, os.getcwd())

import logging
from logger import getlogger
from envs.gui import Renderer
from space_sim.sim import Sim
from envs.sat_gym_env import satGymEnv
from dynamics.static_object import staticObject
from dynamics.dynamic_object import dynamicObject
from dynamics.sat_dynamics import satelliteDynamics
from dynamics.quad_dynamics import quadcopterDynamics
from envs.adversary_train_env import adversaryTrainEnv
from trajectory_planning.path_planner import pathPlanner



logger = getlogger(__name__)

def correct_orbit_units(dynamics):
    orbit_params = {
        'a' : dynamics['a'] << u.km,
        'ecc' : dynamics['ecc'] << u.one,
        'inc' : dynamics['inc'] << u.deg,
        'raan' : dynamics['raan'] << u.deg,
        'argp' : dynamics['argp'] << u.deg,
        'nu' : dynamics['nu'] << u.deg,
    }
    return orbit_params

def make_env(filedir: str, cfg: DictConfig):
    
    orbit_params = correct_orbit_units(cfg['satellite']['dynamics']['initial_orbit'])

    dynamics = satelliteDynamics(
        timestep = cfg['satellite']['dynamics']['timestep'],
        horizon = cfg['satellite']['dynamics']['horizon'],
        pos = np.array(cfg['satellite']['dynamics']['pos']),
        initial_orbit = orbit_params,
        initial_state_data = cfg['satellite']['dynamics']['initial_state_data'],
        spacecraft_data = cfg['satellite']['dynamics']['spacecraft_data'],
        max_control = cfg['satellite']['dynamics']['control_lim']
    )

    satellite = dynamicObject(
        dynamics = dynamics,
        mesh = {'points':np.array(cfg['satellite']['mesh']['points']),'lines':np.array(cfg['satellite']['mesh']['lines'])},
        name = cfg['satellite']['name'],
        pos = np.array(cfg['satellite']['dynamics']['pos']),
    )

    path_planner = pathPlanner(
        goal_state = cfg['path_planner']['goal_state'],
        path_planning_algorithm = cfg['path_planner']['path_planning_algorithm'],
        kwargs = cfg['path_planner']['kwargs'],
        max_distance = cfg['path_planner']['max_distance'],
        interpolation_method = cfg['path_planner']['interpolation_method'],
        n = cfg['path_planner']['n'] 
    )

    kwargs = {}

    for kwarg in cfg['sim']['kwargs'].keys():
        kwargs[kwarg] = cfg['sim']['kwargs'][kwarg]

    sim = Sim(
        main_object = satellite,
        path_planner = path_planner,
        point_cloud_size = cfg['sim']['point_cloud_size'],
        path_point_tolerance = cfg['sim']['path_point_tolerance'],
        point_cloud_radius = cfg['sim']['point_cloud_radius'],
        control_method = cfg['sim']['control_method'],
        goal_tolerance = cfg['sim']['goal_tolerance'],
        kwargs = kwargs,
    )

    if cfg['obstacles'] is None:
        cfg['obstacles'] = []

    if bool(cfg['adversary'][True]):
        logger.info('Initializing adversarial agents')
        for adversary in cfg['adversary']['adversaries']:
            stl = pv.read(filedir+'/'+cfg['adversary']['adversaries'][adversary]['stl'])
            stl.points *= cfg['adversary']['adversaries'][adversary]['stl_scale']

            adversary_dynamics = satelliteDynamics(
                timestep = cfg['satellite']['dynamics']['timestep'],
                horizon = cfg['satellite']['dynamics']['horizon'],
                pos = np.array(cfg['adversary']['adversaries'][adversary]['pos']),
                vel = np.array(cfg['adversary']['adversaries'][adversary]['vel']),
                initial_orbit = orbit_params,
                initial_state_data = cfg['satellite']['dynamics']['initial_state_data'],
                spacecraft_data = cfg['satellite']['dynamics']['spacecraft_data'],
                max_control = cfg['adversary']['adversaries'][adversary]['control_lim'],
            )

            adversary = dynamicObject(
                dynamics = adversary_dynamics, 
                mesh = stl,
                name = cfg['adversary']['adversaries'][adversary]['name'], 
                pos = cfg['adversary']['adversaries'][adversary]['pos']
            )

            kwargs = {}

            for kwarg in cfg['sim']['kwargs'].keys():
                kwargs[kwarg] = cfg['sim']['kwargs'][kwarg]
                
            sim.create_adversary(
                adversary=adversary,
                kwargs=kwargs,
                control_method='MPC'
            )


    
    if bool(cfg['random'][True]):
        for n in range(cfg['random']['n']):
            stl = pv.read(cfg['random']['stl'])
            stl.points *= cfg['random']['stl_scale']

            xlim = cfg['random']['x_range']
            ylim = cfg['random']['y_range']
            zlim = cfg['random']['z_range']
            pos = [np.random.random()*(xlim[1]-xlim[0])+xlim[0], np.random.random()*(ylim[1]-ylim[0])+ylim[0], np.random.random()*(zlim[1]-zlim[0])+zlim[0]]
            vel = np.random.random(3,)
            vel = vel/np.linalg.norm(vel)*(cfg['random']['vel']*np.random.random())

            obs_dynamics = satelliteDynamics(
                timestep = cfg['satellite']['dynamics']['timestep'],
                horizon = cfg['satellite']['dynamics']['horizon'],
                pos = pos,
                vel = vel,
                initial_orbit = orbit_params,
                initial_state_data = cfg['satellite']['dynamics']['initial_state_data'],
                spacecraft_data = cfg['satellite']['dynamics']['spacecraft_data']
            )

            temp_obstacle = dynamicObject(
                dynamics = obs_dynamics, 
                mesh = stl,
                name = 'rand'+str(n), 
                pos = pos)
            
            sim.add_obstacle(obstacle=temp_obstacle)
    else:
        for obstacle in cfg['obstacles']:
            

            stl = pv.read(cfg['obstacles'][obstacle]['stl'])
            stl.points *= cfg['obstacles'][obstacle]['stl_scale']

            obs_dynamics = satelliteDynamics(
                timestep = cfg['satellite']['dynamics']['timestep'],
                horizon = cfg['satellite']['dynamics']['horizon'],
                pos = np.array(cfg['obstacles'][obstacle]['pos']),
                vel = np.array(cfg['obstacles'][obstacle]['vel']),
                initial_orbit = orbit_params,
                initial_state_data = cfg['satellite']['dynamics']['initial_state_data'],
                spacecraft_data = cfg['satellite']['dynamics']['spacecraft_data']
            )

            temp_obstacle = dynamicObject(
                dynamics = obs_dynamics, 
                mesh = stl,
                name = cfg['obstacles'][obstacle]['name'], 
                pos = cfg['obstacles'][obstacle]['pos'])
            
            sim.add_obstacle(obstacle=temp_obstacle)

    env = adversaryTrainEnv(
        sim=sim,
        step_duration=cfg['satellite']['dynamics']['timestep']*cfg['satellite']['dynamics']['horizon'],
        max_episode_length=cfg['env']['max_timestep'],
        max_ctrl=cfg['env']['max_control'],
        action_scaling_type=cfg['env']['action_scaling'],
        randomize_initial_state=cfg['env']['random_initial_state'],
    )


    filter_keys=[
        'sat_state',
        'adversary0_state',
        #'goal_state'
    ]
    env = FilterObservation(
        env,
        filter_keys=filter_keys,
    )
    env = FlattenObservation(env)
    env.unwrapped.seed(cfg["seed"])

    return env

    
@hydra.main(version_base=None, config_path="conf", config_name="config")  
def main(cfg : DictConfig):
    env = make_env(cfg)
    env.reset()
    print(env.action_space)
    print(env.observation_space)
    env.unwrapped.seed()
    for i in range(100):
        obs, rew, done, _ = env.step(np.zeros(len(env.max_ctrl,)))
        #print(obs, rew, done, _)
    env.close()

if __name__ == "__main__":
    main()
    #main()



