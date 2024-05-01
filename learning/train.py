import os
import sys
import hydra
import torch
import logging
import numpy as np
from omegaconf import DictConfig
from stable_baselines3 import PPO
from hydra.core.hydra_config import HydraConfig
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor

from make_env import make_env
from subproc_vec_env_no_daemon import SubprocVecEnvNoDaemon

def mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

@hydra.main(config_path="conf", config_name="config", version_base='1.1')
def train(cfg: DictConfig):
    set_random_seed(cfg["seed"])
    filedir = sys.path[1]
    logdir = os.getcwd() #+ HydraConfig.get().run.dir
    logdir = filedir+logdir[2:]
    if not os.path.exists(filedir+logdir):
        print(logdir)
    print("current directory:", logdir)
    logging.basicConfig(filename=logdir+'\log.log') #set up logger file

    # make parallel envs
    def env_fn(seed):
        env = make_env(filedir,cfg)
        env.seed(seed)
        return env

    class EnvMaker:
        def __init__(self, seed):
            self.seed = seed

        def __call__(self):
            return env_fn(self.seed)


    def make_vec_env(nenvs, seed):
        envs = VecMonitor(
            SubprocVecEnvNoDaemon([EnvMaker(seed + i) for i in range(nenvs)])
        )
        return envs

    env = make_vec_env(cfg["alg"]["nenv"], cfg["seed"])

    # define policy network size
    policy_kwargs = dict(net_arch=dict(pi=cfg["alg"]["pi"], vf=cfg["alg"]["vf"]))
    

    # alg kw for sac or td3
    if cfg["alg"]["type"] == "ppo":
        alg = PPO
        alg_kwargs = {}


    # define model
    model = alg(
            "MlpPolicy", 
            env, 
            verbose=1,
            tensorboard_log=logdir,
            learning_rate=cfg["alg"]["lr"], 
            batch_size=cfg["alg"]["batch_size"],
            gamma=cfg["alg"]["gamma"],
            policy_kwargs=policy_kwargs,
            seed=cfg["seed"],
            device=cfg["alg"]["device"],
            **alg_kwargs,
            n_steps=10,
            )


    # train
    model.learn(
            total_timesteps=cfg["alg"]["total_timesteps"],
            progress_bar=True,
            )


    # end
    env.close()
    model.save(os.path.join(logdir, "model"))


if __name__ == "__main__":
    torch.set_num_threads(9)
    sys.path.insert(1, os.getcwd())
    train()