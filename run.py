import os
import ray
import hydra
import torch
from omegaconf import DictConfig
from hydra.utils import get_original_cwd

from logger import getlogger
from learning.train_ray import train_ray
from learning.train_sb3 import train, retrain
from learning.run_space_sim import runSpaceSim

DIRECTORY = None

@hydra.main(config_path="learning/conf", config_name="config2", version_base='1.1')
def train_rl_model(cfg: DictConfig):
    if 'ray' in cfg['alg']['lib']:
        ray.init(runtime_env={'working_dir': '/home/cameron/magpie_rl',
                              'env_vars': {'PYTHONWARNINGS': 'ignore::DeprecationWarning'}})
        train_ray(cfg,DIRECTORY)
    else:
        os.chdir('../../../')
        train(cfg,DIRECTORY)

@hydra.main(config_path="learning/conf", config_name="config1", version_base='1.1')
def retrain_rl_model(cfg: DictConfig):
    modeldir = '/home/cameron/magpie_rl/logs/evade_1m/2024-06-01/model.zip'
    retrain(cfg,DIRECTORY,modeldir)

@hydra.main(config_path="learning/conf", config_name="config1", version_base='1.1')
def run_rl_model(cfg: DictConfig):
    os.chdir('../../../')
    modeldir = '/home/cameron/magpie_rl/logs/img3d_test_1024_5obs/2024-08-02/model.zip'
    runSpaceSim(cfg,DIRECTORY,modeldir=modeldir,render=False)

if __name__ == "__main__":
    torch.set_num_threads(9)
    DIRECTORY = os.getcwd()
    run_rl_model()
    