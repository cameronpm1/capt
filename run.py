import os
import ray
import hydra
import torch
from omegaconf import DictConfig
from hydra.utils import get_original_cwd

from logger import getlogger
from learning.train_ray import train_ray
from learning.train_sb3 import train, retrain
from learning.run_space_sim_ray import runSpaceSimRay
from learning.run_space_sim_sb3 import runSpaceSimSb3

DIRECTORY = None

@hydra.main(config_path="learning/conf", config_name="config1", version_base='1.1')
def train_rl_model(cfg: DictConfig):
    if 'ray' in cfg['alg']['lib']:
        ray.init(runtime_env={'working_dir': '/home/cameron/magpie_rl',
                              'env_vars': {'PYTHONWARNINGS': 'ignore::DeprecationWarning'}})
        train_ray(cfg,DIRECTORY)
    else:
        os.chdir('../../../')
        train(cfg,DIRECTORY)

@hydra.main(config_path="learning/conf", config_name="config2", version_base='1.1')
def retrain_rl_model(cfg: DictConfig):
    modeldir = '/home/cameron/magpie_rl/logs/evade_1m/2024-06-01/model.zip'
    retrain(cfg,DIRECTORY,modeldir)

@hydra.main(config_path="learning/conf", config_name="config2", version_base='1.1')
def run_rl_model(cfg: DictConfig):
    os.chdir('../../../')
    modeldir = '/home/cameron/magpie_rl/logs/test18/2024-08-07/checkpoint5353200'
    modeldir = '/home/cameron/magpie_rl/logs/2d_flat/2024-08-08/checkpoint13442400/policies/policy0'
    if 'ray' in cfg['alg']['lib']:
        runSpaceSimRay(cfg,DIRECTORY,modeldir=modeldir,render=False)
    else:
        runSpaceSimSb3(cfg,DIRECTORY,modeldir=modeldir,render=False)

if __name__ == "__main__":
    torch.set_num_threads(8)
    DIRECTORY = os.getcwd()
    train_rl_model()
    