import os,sys
#sys.path.append('c:/Users/Cameron Mehlman/Documents/Magpie')

import time
import hydra
import numpy as np
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.rllib.policy.policy import Policy
from omegaconf import DictConfig, OmegaConf
from ray.rllib.algorithms.algorithm import Algorithm

from envs.gui import Renderer
from learning.make_env import make_env
from custom_model_archs.sirenfcnet import SirenOutFullyConnectedNetwork

ModelCatalog.register_custom_model("sirenfcnet", SirenOutFullyConnectedNetwork)

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

    if render:
        renderer = Renderer(
            xlim = [-50,50],
            ylim = [-50,50],
            zlim = [-50,50],
            vista = False,
            dim = env.unwrapped.dim,
        )

    first_goal_count = 0
    second_goal_count = 0
    eps = 50

    for i in range(eps):

        obs, _ = env.reset()

        if render:
            renderer.plot(env.render())

        if modeldir is not None:
            if cfg['alg']['type'] == 'sac':
                #algo = Algorithm.from_checkpoint(modeldir)
                #model = lambda obs: algo.build().get_policy().compute_single_action(obs)
                policy = Policy.from_checkpoint(modeldir)
                model = lambda obs: policy.compute_single_action(obs)


        time.sleep(10)
        timesteps = cfg['env']['max_timestep']
        timestep = 0
        done = False

        for i in range(timesteps):
            if modeldir is None:
                action = np.zeros(env.unwrapped.dim,)
            else:
                action,_,_ = model(obs)
            obs,rewards,terminated,truncated,info = env.step(action)
            timestep += 1
            #print(env.unwrapped.sim.main_object.dynamics.get_pos())
            #print(env.unwrapped.sim.adversary[0].dynamics.get_pos(),env.unwrapped.sim.adversary[1].dynamics.get_pos())
            if i%10 == 0:
                if verbose:
                    print('at timestep',i,'distance to goal:', env.unwrapped.sim.distance_to_goal())
            if terminated or truncated:
                if 'obs' in cfg['env']['scenario']:
                    if info['success']:
                        first_goal_count += 1
                else:
                    timestep = i
                    if info['goal_count'] == 1:
                        first_goal_count += 1
                    if info['goal_count'] == 2:
                        second_goal_count += 1
                break
            if render:
                renderer.plot(env.unwrapped.render())

        #print('Simulation ended at timestep',timestep)
    
    print('total episodes:',eps)
    print('first goal reached:',first_goal_count)
    print('second goal reached:',second_goal_count)

    return first_goal_count, second_goal_count, eps

if __name__ == "__main__":
    sys.path.insert(1, os.getcwd())

    model_dir = None

    runSpaceSim(modeldir=model_dir)