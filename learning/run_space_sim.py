import os,sys
sys.path.append('c:/Users/Cameron Mehlman/Documents/Magpie')

import time
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from make_env import make_env
from envs.gui import Renderer

def simulate_space_env(model_dir, verbose=True) -> None:

    @hydra.main(version_base=None, config_path="conf", config_name="config")
    def simulate(cfg : DictConfig):
        
        filedir = sys.path[1]
        env = make_env(filedir,cfg)
        env.reset()

        renderer = Renderer(
            xlim = [-10,10],
            ylim = [-5,30],
            zlim = [-10,10],
            vista = False,
        )

        renderer.plot(env.render())

        time.sleep(10)
        timesteps = 1500
        timestep = 0
        done = False

        for i in range(timesteps):
            if model_dir is None:
                action = np.zeros(9,)
            _,_,_,_,_ = env.step(action)
            if i%15 == 0:
                if verbose:
                    print('at timestep',i,'distance to goal:', env.unwrapped.sim.get_distance_to_goal())
            if done:
                timestep = i
                break
            renderer.plot(env.render())


        print('Simulation ended at timestep',timestep)

    simulate()

if __name__ == "__main__":
    sys.path.insert(1, os.getcwd())

    model_dir = None

    simulate_space_env(model_dir=model_dir)