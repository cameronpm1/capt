import os
import hydra
import torch
from omegaconf import DictConfig

from learning.train import train

DIRECTORY = None

@hydra.main(config_path="learning/conf", config_name="config", version_base='1.1')
def train_rl_model(cfg: DictConfig):
    train(cfg,DIRECTORY)

if __name__ == "__main__":
    torch.set_num_threads(9)
    DIRECTORY = os.getcwd()
    train_rl_model()
    