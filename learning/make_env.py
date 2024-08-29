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
from envs.marl_env import MARLEnv
from envs.sat_gym_env import satGymEnv
from envs.marl_test_env import MARLTestEnv
from dynamics.twod_dynamics import twodDynamics
from dynamics.static_object import staticObject
from dynamics.dynamic_object import dynamicObject
from dynamics.sat_dynamics import satelliteDynamics
from dynamics.quad_dynamics import quadcopterDynamics
from envs.controler_train_env import controlerTrainEnv
from trajectory_planning.path_planner import pathPlanner
from envs.controler_train_env_image import controlerTrainEnvImage



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

    def initialize_obstacles(sim):
        if bool(cfg['random'][True]):
            logger.info('Initializing random obstacles')
            for n in range(cfg['random']['n']):
                xlim = cfg['random']['x_range']
                ylim = cfg['random']['y_range']
                zlim = cfg['random']['z_range']
                pos = [np.random.random()*(xlim[1]-xlim[0])+xlim[0], np.random.random()*(ylim[1]-ylim[0])+ylim[0], np.random.random()*(zlim[1]-zlim[0])+zlim[0]]
                pos = [1000.0,1000.0,1000.0] #initialize obstacles off of the board
                vel = np.random.random(3,)
                vel = vel/np.linalg.norm(vel)*(cfg['random']['vel']*np.random.random())

                if cfg['env']['dim'] == 2:
                    obs_dynamics = twodDynamics(
                        timestep = cfg['satellite']['dynamics']['timestep'],
                        horizon = cfg['satellite']['dynamics']['horizon'],
                        pos = pos[0:2],
                        vel = vel[0:2],
                        euler = [0],
                        data = cfg['satellite']['dynamics']['data'],
                    )

                    points = []
                    for i in range(int(cfg['sim']['point_cloud_size']/cfg['random']['n'])):
                        vec = np.random.random((2,))
                        points.append((vec/np.linalg.norm(vec)))

                    temp_obstacle = dynamicObject(
                        dynamics = obs_dynamics, 
                        mesh = {'points':np.array(points),'lines':np.array([])},
                        name = 'obstacle'+str(n), 
                        pos = pos[0:2])
                    
                else:
                    obs_dynamics = satelliteDynamics(
                        timestep = cfg['satellite']['dynamics']['timestep'],
                        horizon = cfg['satellite']['dynamics']['horizon'],
                        pos = pos,
                        vel = vel,
                        spacecraft_data = cfg['satellite']['dynamics']['spacecraft_data']
                    )

                    points = []
                    for i in range(int(cfg['sim']['point_cloud_size']/cfg['random']['n'])):
                        vec = np.random.random((3,))
                        points.append((vec/np.linalg.norm(vec)))

                    temp_obstacle = dynamicObject(
                        dynamics = obs_dynamics, 
                        mesh = {'points':np.array(points),'lines':np.array([])},
                        name = 'obstacle'+str(n), 
                        pos = pos[0:3])

                sim.add_obstacle(obstacle=temp_obstacle)

    def initialize_adversasries(sim):
        if bool(cfg['adversary'][True]):
            logger.info('Initializing adversarial agents')
            for adversary in cfg['adversary']['adversaries']:

                if cfg['env']['dim'] == 2:
                    stl = {
                        'points':np.array(cfg['adversary']['adversaries'][adversary]['mesh']['points']),
                        'lines':np.array(cfg['adversary']['adversaries'][adversary]['mesh']['lines'])
                        }

                    adversary_dynamics = twodDynamics(
                        timestep = cfg['satellite']['dynamics']['timestep'],
                        horizon = cfg['satellite']['dynamics']['horizon'],
                        pos = np.array(cfg['adversary']['adversaries'][adversary]['pos']),
                        vel = np.array(cfg['adversary']['adversaries'][adversary]['vel']),
                        euler = np.array(cfg['adversary']['adversaries'][adversary]['euler']),
                        data = cfg['adversary']['adversaries'][adversary]['data'],
                        max_control = cfg['adversary']['adversaries'][adversary]['control_lim']
                    )
                else:
                    stl = pv.read(filedir+'/'+cfg['adversary']['adversaries'][adversary]['stl'])
                    stl.points *= cfg['adversary']['adversaries'][adversary]['stl_scale']

                    adversary_dynamics = satelliteDynamics(
                        timestep = cfg['satellite']['dynamics']['timestep'],
                        horizon = cfg['satellite']['dynamics']['horizon'],
                        pos = np.array(cfg['adversary']['adversaries'][adversary]['pos']),
                        vel = np.array(cfg['adversary']['adversaries'][adversary]['vel']),
                        initial_orbit = orbit_params,
                        spacecraft_data = cfg['satellite']['dynamics']['spacecraft_data'],
                        max_control = cfg['adversary']['adversaries'][adversary]['control_lim'],
                    )

                
                adversary = dynamicObject(
                    dynamics = adversary_dynamics, 
                    mesh = stl,
                    name = cfg['adversary']['adversaries'][adversary]['name'], 
                    pos = cfg['adversary']['adversaries'][adversary]['pos'],
                    dim = cfg['env']['dim'],
                )
     
                sim.create_adversary(
                    adversary=adversary,
                )

    if cfg['env']['dim'] == 2:
        dynamics = twodDynamics(
            timestep = cfg['satellite']['dynamics']['timestep'],
            horizon = cfg['satellite']['dynamics']['horizon'],
            pos = np.array(cfg['satellite']['dynamics']['pos']),
            vel = np.array(cfg['satellite']['dynamics']['vel']),
            euler = np.array(cfg['satellite']['dynamics']['euler']),
            data = cfg['satellite']['dynamics']['data'],
            max_control = cfg['satellite']['dynamics']['control_lim']
        )
    else:
        if 'initial_orbit' in cfg['satellite']['dynamics'].keys():
            orbit_params = correct_orbit_units(cfg['satellite']['dynamics']['initial_orbit'])
        else:
            orbit_params = None

        dynamics = satelliteDynamics(
            timestep = cfg['satellite']['dynamics']['timestep'],
            horizon = cfg['satellite']['dynamics']['horizon'],
            pos = np.array(cfg['satellite']['dynamics']['pos']),
            vel = np.array(cfg['satellite']['dynamics']['vel']),
            initial_orbit = orbit_params,
            spacecraft_data = cfg['satellite']['dynamics']['spacecraft_data'],
            max_control = cfg['satellite']['dynamics']['control_lim']
        )
        

    satellite = dynamicObject(
        dynamics = dynamics,
        mesh = {'points':np.array(cfg['satellite']['mesh']['points']),'lines':np.array(cfg['satellite']['mesh']['lines'])},
        name = cfg['satellite']['name'],
        pos = np.array(cfg['satellite']['dynamics']['pos']),
        dim = cfg['env']['dim'],
    )

    path_planner_kwargs = cfg['path_planner']['kwargs']
    if 'test' in cfg['env']['scenario'] and cfg['env']['evader_policy_dir'] is None:
        #histogram does not work w evade, must overwrite if evade is to be used
        path_planner_kwargs['data_format'] = 'grid'
        

    path_planner = pathPlanner(
        goal_state = cfg['path_planner']['goal_state'],
        path_planning_algorithm = cfg['path_planner']['path_planning_algorithm'],
        kwargs = cfg['path_planner']['kwargs'],
        max_distance = cfg['path_planner']['max_distance'],
        interpolation_method = cfg['path_planner']['interpolation_method'],
        n = cfg['path_planner']['n'],
        dim = cfg['env']['dim'],
    )

    kwargs = {}

    for kwarg in cfg['sim']['kwargs'].keys():
        kwargs[kwarg] = cfg['sim']['kwargs'][kwarg]

    if 'ray' in cfg['alg']['lib']:
        parallel_envs = cfg['alg']['nenv']*cfg['alg']['cpu_envs']
    else:
        parallel_envs = cfg['alg']['nenv']

    if 'marl' in cfg['env']['scenario']:

        '''
        only add obstacles and adversaries if the word adversary is in the env scenario name
        '''

        sim = Sim(
            main_object = satellite,
            path_planner = path_planner,
            point_cloud_size = cfg['sim']['point_cloud_size'],
            path_point_tolerance = cfg['sim']['path_point_tolerance'],
            point_cloud_radius = cfg['sim']['point_cloud_radius'],
            control_method = cfg['sim']['control_method'],
            goal_tolerance = cfg['sim']['goal_tolerance'],
            collision_tolerance = cfg['sim']['collision_tolerance'],
            track_point_cloud = cfg['sim']['track_point_cloud'],
            kwargs = kwargs,
        )

        logger.info('Setting up evasion environment')

        initialize_adversasries(sim)
        initialize_obstacles(sim)

        env = MARLEnv(
            sim=sim,
            step_duration=cfg['satellite']['dynamics']['timestep']*cfg['satellite']['dynamics']['horizon'],
            max_episode_length=cfg['env']['max_timestep'],
            sat_max_ctrl=cfg['env']['sat_max_control'],
            adv_max_ctrl=cfg['env']['adv_max_control'],
            total_train_steps=cfg['alg']['total_timesteps'],
            action_scaling_type=cfg['env']['action_scaling'],
            randomize_initial_state=cfg['env']['random_initial_state'],
        )

    elif 'controlImage' in cfg['env']['scenario']:
        '''
        set up controlImage env
        '''

        sim = Sim(
            main_object = satellite,
            path_planner = path_planner,
            point_cloud_size = cfg['sim']['point_cloud_size'],
            path_point_tolerance = cfg['sim']['path_point_tolerance'],
            point_cloud_radius = cfg['sim']['point_cloud_radius'],
            control_method = cfg['sim']['control_method'],
            goal_tolerance = cfg['sim']['goal_tolerance'],
            collision_tolerance = cfg['sim']['collision_tolerance'],
            track_point_cloud = cfg['sim']['track_point_cloud'],
            kwargs = kwargs,
        )

        initialize_obstacles(sim)

        logger.info('Initializing control environment')
        env = controlerTrainEnvImage(
            sim=sim,
            step_duration=cfg['satellite']['dynamics']['timestep']*cfg['satellite']['dynamics']['horizon'],
            max_episode_length=cfg['env']['max_timestep'],
            max_ctrl=cfg['env']['sat_max_control'],
            total_train_steps=cfg['alg']['total_timesteps'],
            action_scaling_type=cfg['env']['action_scaling'],
            randomize_initial_state=cfg['env']['random_initial_state'],
            parallel_envs=parallel_envs,
            curriculum=cfg['env']['curriculum']
        )


        filter_keys=[
            'rel_goal_state',
            'obstacles_matrix'
        ]

        env = FilterObservation(env,filter_keys=filter_keys)
        env = FlattenObservation(env)

    elif 'test' in cfg['env']['scenario']:
        '''
        set up test env
        '''

        sim = Sim(
            main_object = satellite,
            path_planner = path_planner,
            point_cloud_size = cfg['sim']['point_cloud_size'],
            path_point_tolerance = cfg['sim']['path_point_tolerance'],
            point_cloud_radius = cfg['sim']['point_cloud_radius'],
            control_method = cfg['sim']['control_method'],
            goal_tolerance = cfg['sim']['goal_tolerance'],
            collision_tolerance = cfg['sim']['collision_tolerance'],
            track_point_cloud = cfg['sim']['track_point_cloud'],
            kwargs = kwargs,
        )

        initialize_adversasries(sim)
        initialize_obstacles(sim)

        logger.info('Initializing control environment')
        env = MARLTestEnv(
            sim=sim,
            step_duration=cfg['satellite']['dynamics']['timestep']*cfg['satellite']['dynamics']['horizon'],
            max_episode_length=cfg['env']['max_timestep'],
            sat_max_ctrl=cfg['env']['sat_max_control'],
            adv_max_ctrl=cfg['env']['adv_max_control'],
            action_scaling_type=cfg['env']['action_scaling'],
            evader_policy_dir=cfg['env']['evader_policy_dir'],
            randomize_initial_state=cfg['env']['random_initial_state'],
        )

    else:
        '''
        set up control env
        '''

        sim = Sim(
            main_object = satellite,
            path_planner = path_planner,
            point_cloud_size = cfg['sim']['point_cloud_size'],
            path_point_tolerance = cfg['sim']['path_point_tolerance'],
            point_cloud_radius = cfg['sim']['point_cloud_radius'],
            control_method = cfg['sim']['control_method'],
            goal_tolerance = cfg['sim']['goal_tolerance'],
            collision_tolerance = cfg['sim']['collision_tolerance'],
            track_point_cloud = cfg['sim']['track_point_cloud'],
            kwargs = kwargs,
        )

        initialize_obstacles(sim)

        logger.info('Initializing control environment')
        env = controlerTrainEnv(
            sim=sim,
            step_duration=cfg['satellite']['dynamics']['timestep']*cfg['satellite']['dynamics']['horizon'],
            max_episode_length=cfg['env']['max_timestep'],
            max_ctrl=cfg['env']['sat_max_control'],
            total_train_steps=cfg['alg']['total_timesteps'],
            action_scaling_type=cfg['env']['action_scaling'],
            randomize_initial_state=cfg['env']['random_initial_state'],
            parallel_envs=parallel_envs,
        )

        filter_keys=[
            'rel_goal_state'
        ]

        if bool(cfg['random'][True]):
            for n in range(cfg['random']['n']):
                obs_key = 'rel_obstacle'+str(n)+'_state'
                filter_keys.append(obs_key)

        env = FilterObservation(env,filter_keys=filter_keys)
        env = FlattenObservation(env)

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



