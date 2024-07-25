import os
import ray
import hydra
import torch
from omegaconf import DictConfig

from logger import getlogger
from learning.train_ray import train_ray
from learning.train_sb3 import train, retrain
from learning.run_space_sim import runSpaceSim

DIRECTORY = None

@hydra.main(config_path="learning/conf", config_name="config2", version_base='1.1')
def train_rl_model(cfg: DictConfig):
    if 'ray' in cfg['alg']['lib']:
        ray.init(runtime_env={'working_dir': r'C:\Users\Cameron Mehlman\Documents\magpie_rl',
                              'env_vars': {'PYTHONWARNINGS': 'ignore::DeprecationWarning'}})
        train_ray(cfg,DIRECTORY)
    else:
        train(cfg,DIRECTORY)

@hydra.main(config_path="learning/conf", config_name="config1", version_base='1.1')
def retrain_rl_model(cfg: DictConfig):
    modeldir = '/home/cameron/magpie_rl/logs/evade_1m/2024-06-01/model.zip'
    retrain(cfg,DIRECTORY,modeldir)

@hydra.main(config_path="learning/conf", config_name="config2", version_base='1.1')
def run_rl_model(cfg: DictConfig):
    #modeldir = '/home/cameron/magpie_rl/logs/adversary_rew2/2024-05-28/midtrain_model_17000000_steps.zip' #adversary model
    #modeldir = '/home/cameron/magpie_rl/logs/evade_1m/2024-06-01/retrained_model.zip' # evade model
    modeldir = None
    #modeldir = '/home/cameron/magpie_rl/logs/control/2024-05-21/retrained_model.zip' #control model dir
    runSpaceSim(cfg,DIRECTORY,modeldir=modeldir,render=True)

if __name__ == "__main__":
    torch.set_num_threads(12)
    DIRECTORY = os.getcwd()
    train_rl_model()
    