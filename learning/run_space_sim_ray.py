import os,sys
#sys.path.append('c:/Users/Cameron Mehlman/Documents/Magpie')

import time
import hydra
import numpy as np
from ray.tune.registry import register_env
from ray.rllib.policy.policy import Policy
from omegaconf import DictConfig, OmegaConf
from ray.rllib.algorithms.algorithm import Algorithm

from envs.gui import Renderer
from learning.make_env import make_env

def runSpaceSimRay(
        cfg : DictConfig, 
        filedir,
        modeldir=None, 
        render=False, 
        verbose=True
    ) -> None:

    filedir = filedir
    env = make_env(filedir,cfg)
    env.unwrapped.seed(seed=cfg['seed'])

    #make env function 
    def env_maker(config):
        str(config) #needed to call worker and vector _index (odd bug w rllib)
        env = make_env(filedir,cfg)
        #if config has parameters, ensure envs have different seeds
        seed = cfg['seed']
        if len(config.keys()) > 0:
            seed += ((100*config.worker_index) + config.vector_index)
        env.unwrapped.seed(seed)
        return env

    def multi_env_maker(config):
        #[env_maker({},seed=(cfg['seed'] + (i*100))) for i in range(#cfg['env']['nenvs'])]
        env = env_maker(config)  
        vec_env = multiEnvWrapper(env)
        return vec_env
    
    if 'multi' in cfg['env']['scenario']:
        env_name = cfg['env']['scenario']
        register_env(env_name, multi_env_maker) #register make env function 
        #test_env for getting obs/action space
        policy_list = ['policy'+str(i) for i in range(cfg['env']['n_policies'])]
    else:
        env_name = cfg['env']['scenario']
        register_env(env_name, env_maker) #register make env function
        #test_env for getting obs/action space

    for i in range(1):

        obs, _ = env.reset()

        if render:
            renderer = Renderer(
                xlim = [-10,10],
                ylim = [-5,30],
                zlim = [-10,10],
                vista = False,
                dim = env.unwrapped.dim,
            )
            renderer.plot(env.render())

        if modeldir is not None:
            if cfg['alg']['type'] == 'sac':
                #algo = Algorithm.from_checkpoint(modeldir)
                #model = lambda obs: algo.build().get_policy().compute_single_action(obs)
                policy = Policy.from_checkpoint(modeldir)
                model = lambda obs: policy.compute_single_action(obs)


        time.sleep(10)
        timesteps = 1500
        timestep = 0
        done = False

        for i in range(timesteps):
            if modeldir is None:
                action = np.zeros(env.unwrapped.dim,)
            else:
                action,_,_ = model(obs)
            obs,rewards,terminated,truncated,_ = env.step(action)
            if i%10 == 0:
                if verbose:
                    print('at timestep',i,'distance to goal:', env.unwrapped.sim.distance_to_goal())
            if terminated or truncated:
                timestep = i
                break
            if render:
                renderer.plot(env.unwrapped.render())

        print('Simulation ended at timestep',timestep)
    

if __name__ == "__main__":
    sys.path.insert(1, os.getcwd())

    model_dir = None

    runSpaceSim(modeldir=model_dir)