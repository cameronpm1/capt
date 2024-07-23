import os
import sys
import hydra
import torch
import logging
import numpy as np
from omegaconf import DictConfig
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from hydra.core.hydra_config import HydraConfig
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor


from logger import getlogger
from learning.make_env import make_env
from learning.subproc_vec_env_no_daemon import SubprocVecEnvNoDaemon

logger = getlogger(__name__)

def mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)


def train(cfg: DictConfig,filedir):
    filedir = filedir
    set_random_seed(cfg['seed'])
    logdir = cfg['logging']['run']['dir']
    logdir = filedir+logdir
    if not os.path.exists(logdir):
        logger.info("Safe directory not found, creating path ...")
        mkdir(logdir)
    else:
        logger.info("Save directory found ...")
    print("current directory:", logdir)
    #logging.basicConfig(filename=logdir+'\log.log') #set up logger file

    # make parallel envs
    def env_fn(seed):
        env = make_env(filedir,cfg)
        env.unwrapped.seed(seed)
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

    logger.info("Preparing to initialize parallel environments ...")
    env = make_vec_env(cfg["alg"]["nenv"], cfg["seed"])
    logger.info("Parallel environments initialized ...")
    # define policy network size
    
    

    # alg kw for sac or td3
    if cfg["alg"]["type"] == "ppo":
        alg = PPO
        policy_kwargs = dict(net_arch=dict(pi=cfg["alg"]["pi"], vf=cfg["alg"]["vf"]))
        alg_kwargs = {}

    if cfg["alg"]["type"] == "sac":
        alg = SAC
        policy_kwargs = dict(net_arch=dict(pi=cfg["alg"]["pi"], qf=cfg["alg"]["vf"]))
        alg_kwargs = {}

    logger.info("Initializing model ...")
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
            )

    checkpoint_callback = CheckpointCallback(
            save_freq=50000,
            save_path=logdir,
            name_prefix="midtrain_model",
            save_replay_buffer=True,
            save_vecnormalize=True,
            )


    logger.info("Beginning training ...")
    # train
    model.learn(
            total_timesteps=cfg["alg"]["total_timesteps"],
            progress_bar=True,
            callback=checkpoint_callback,
            )


    # end
    env.close()
    logger.info("Saving model ...")
    model.save(os.path.join(logdir, "model"))

def retrain(cfg: DictConfig,filedir,modeldir):
    filedir = filedir
    set_random_seed(cfg['seed'])
    logdir = cfg['logging']['run']['dir']
    logdir = filedir+logdir
    if not os.path.exists(logdir):
        logger.info("Safe directory not found, creating path ...")
        mkdir(logdir)
    else:
        logger.info("Save directory found ...")
    print("current directory:", logdir)
    #logging.basicConfig(filename=logdir+'\log.log') #set up logger file

    # make parallel envs
    def env_fn(seed):
        env = make_env(filedir,cfg)
        env.unwrapped.seed(seed)
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

    logger.info("Preparing to initialize parallel environments ...")
    env = make_vec_env(cfg["alg"]["nenv"], cfg["seed"])
    logger.info("Parallel environments initialized ...")
    # define policy network size
    

    if cfg["alg"]["type"] == "ppo":
        alg = PPO
        logger.info("Loading model ...")
        model = PPO.load(modeldir)
    elif cfg["alg"]["type"] == "sac":
        alg = SAC
        logger.info("Loading model ...")
        model = SAC.load(modeldir)


    model.tensorboard_log = logdir
    logger.info("Loading parallel environments ...")
    # load env
    model.set_env(env)

    logger.info("Loading model ...")
    # load model
    model = PPO.load(modeldir)
    model.tensorboard_log = logdir

    logger.info("Loading parallel environments ...")
    # load env
    model.set_env(env)

    checkpoint_callback = CheckpointCallback(
            save_freq=50000,
            save_path=logdir,
            name_prefix="retrain_model",
            save_replay_buffer=True,
            save_vecnormalize=True,
            )


    logger.info("Beginning training ...")
    # train
    model.learn(
            total_timesteps=cfg["alg"]["total_timesteps"],
            progress_bar=True,
            callback=checkpoint_callback,
            )


    # end
    env.close()
    logger.info("Saving model ...")
    model.save(os.path.join(logdir, "retrained_model"))



if __name__ == "__main__":
    torch.set_num_threads(9)
    DIRECTORY = os.getcwd()
    train()