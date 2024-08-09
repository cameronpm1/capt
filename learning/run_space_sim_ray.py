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
                    dists = []
                    for i in range(20):
                        dist = env.unwrapped.sim.distance_to_obstacle(idx=i)
                        dists.append(dist)
                    print(min(dists))
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