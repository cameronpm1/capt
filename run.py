import os
import hydra
import torch
from omegaconf import DictConfig

from learning.train import train, retrain
from learning.run_space_sim import runSpaceSim

DIRECTORY = None

@hydra.main(config_path="learning/conf", config_name="config", version_base='1.1')
def train_rl_model(cfg: DictConfig):
    train(cfg,DIRECTORY)

@hydra.main(config_path="learning/conf", config_name="config", version_base='1.1')
def retrain_rl_model(cfg: DictConfig):
    #modeldir = '/home/cameron/magpie_rl/logs/adversary_test/2024-05-13/model.zip'
    modeldir = '/home/cameron/magpie_rl/logs/control/2024-05-21/model.zip'
    retrain(cfg,DIRECTORY,modeldir)

@hydra.main(config_path="learning/conf", config_name="config", version_base='1.1')
def run_rl_model(cfg: DictConfig):
    modeldir = '/home/cameron/magpie_rl/logs/adversary_test/2024-05-13/model.zip'
    #modeldir = '/home/cameron/magpie_rl/logs/control/2024-05-21/retrained_model.zip' #control model dir
    runSpaceSim(cfg,DIRECTORY,modeldir=modeldir,render=False)

if __name__ == "__main__":
    torch.set_num_threads(9)
    DIRECTORY = os.getcwd()
    run_rl_model()
    