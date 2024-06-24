import os,sys
#sys.path.append('c:/Users/Cameron Mehlman/Documents/Magpie')

import time
import hydra
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from omegaconf import DictConfig, OmegaConf

from envs.gui import Renderer
from learning.make_env import make_env

def runSpaceSim(
        cfg : DictConfig, 
        filedir,
        modeldir=None, 
        render=False, 
        verbose=True
    ) -> None:

    filedir = filedir
    env = make_env(filedir,cfg)

    for i in range(1):

        obs, _ = env.reset()

        if render:
            renderer = Renderer(
                xlim = [-10,10],
                ylim = [-5,30],
                zlim = [-10,10],
                vista = False,
            )
            renderer.plot(env.render())

        if modeldir is not None:
            if cfg["alg"]["type"] == "ppo":
                model = PPO.load(modeldir)
            if cfg["alg"]["type"] == "sac":
                model = SAC.load(modeldir)


        time.sleep(10)
        timesteps = 1500
        timestep = 0
        done = False

        for i in range(timesteps):
            if modeldir is None:
                action = np.zeros(3,)
            else:
                action, _states = model.predict(obs)
            obs,rewards,terminated,truncated,_ = env.step(action)
            if i%15 == 0:
                if verbose:
                    print('at timestep',i,'distance to goal:', env.unwrapped.sim.distance_to_goal())
            if terminated or truncated:
                timestep = i
                break
            if render:
                renderer.plot(env.render())

        print('Simulation ended at timestep',timestep)
    

if __name__ == "__main__":
    sys.path.insert(1, os.getcwd())

    model_dir = None

    runSpaceSim(modeldir=model_dir)