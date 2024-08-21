import os
import ray
import time
import hydra
import torch
import argparse
from pathlib import Path
from typing import Optional
from omegaconf import OmegaConf
from omegaconf import DictConfig
from hydra.utils import get_original_cwd
from hydra.experimental import compose, initialize

from logger import getlogger
from learning.train_ray import train_ray
from learning.train_sb3 import train, retrain
from learning.run_space_sim_ray import runSpaceSimRay
from learning.run_space_sim_sb3 import runSpaceSimSb3
from plotting.density_plot import action_density_plot
from plotting.density_plot import collect_action_dist_data

DIRECTORY = None
CONFIG_FILE = None

def remove_argument(parser, arg):
    for action in parser._actions:
        opts = action.option_strings
        if (opts and opts[0] == arg) or action.dest == arg:
            parser._remove_action(action)
            break

    for action in parser._action_groups:
        for group_action in action._group_actions:
            opts = group_action.option_strings
            if (opts and opts[0] == arg) or group_action.dest == arg:
                action._group_actions.remove(group_action)
                return
            
def remove_options(parser, options):
    for option in options:
        for action in parser._actions:
            if vars(action)['option_strings'][0] == option:
                parser._handle_conflict_resolve(None,[(option,action)])
                break

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=str, default='train')
    parser.add_argument('--env', type=str, default='evade_obs')
    parser.add_argument('--dim', type=int, default=2)

    args = parser.parse_args()

    return args

#@hydra.main(config_path=".", config_name=CONFIG_FILE, version_base=None)
def train_rl_model(cfg: DictConfig):
    if 'ray' in cfg['alg']['lib']:
        ray.init(runtime_env={'working_dir': '/home/cameron/magpie_rl',
                              'env_vars': {'PYTHONWARNINGS': 'ignore::DeprecationWarning'},
                              'excludes': ['.git/',]
                              })
        train_ray(cfg,DIRECTORY)
    else:
        os.chdir('../../../')
        train(cfg,DIRECTORY)

#@hydra.main(config_path="learning/conf", config_name=CONFIG_FILE, version_base='1.1')
def retrain_rl_model(cfg: DictConfig):
    modeldir = '/home/cameron/magpie_rl/logs/evade_1m/2024-06-01/model.zip'
    retrain(cfg,DIRECTORY,modeldir)

#@hydra.main(config_path="learning/conf", config_name=CONFIG_FILE, version_base='1.1')
def run_rl_model(cfg: DictConfig):
    os.chdir('../../../')
    modeldir = '/home/cameron/magpie_rl/logs/test18/2024-08-07/checkpoint5353200'
    modeldir = '/home/cameron/magpie_rl/logs/2d_flat/2024-08-08/checkpoint13442400/policies/policy0'
    if 'ray' in cfg['alg']['lib']:
        if 'test' in cfg['env']['scenario']:
            modeldir=None
        runSpaceSimRay(cfg,DIRECTORY,modeldir=modeldir,render=True)
    else:
        runSpaceSimSb3(cfg,DIRECTORY,modeldir=modeldir,render=False)

    return modeldir

#@hydra.main(config_path="learning/conf", config_name=CONFIG_FILE, version_base='1.1')
def collect_data(cfg: DictConfig):
    os.chdir('../../../')
    master_dir = '/home/cameron/magpie_rl/logs/div_policies3'
    master_dir = '/home/cameron/magpie_rl/logs/nondiv_policies3'
    collect_action_dist_data(cfg,DIRECTORY,master_dir)

if __name__ == "__main__":
    '''
    example script:

    python run.py --run train --env divadversary --dim 2
    
    '''

    #torch.set_num_threads(8)
    DIRECTORY = os.getcwd()
    args = get_args()
    dim_end = str(args.dim) + 'd'

    if 'evade' in args.env:
        if 'obs' in args.env:
            CONFIG_FILE = 'obstacle_evader_config' + dim_end
        else:
            CONFIG_FILE = 'adversary_evader_config' + dim_end
    elif 'adversary' in args.env:
        if 'div' in args.env:
            CONFIG_FILE = 'divergent_adversary_config' + dim_end
        else:
            CONFIG_FILE = 'adversary_config' + dim_end
    elif 'control' in args.env:
        CONFIG_FILE = 'controller_config' + dim_end
    elif 'marl' in args.env:
        if 'base' in args.env:
            CONFIG_FILE = 'marl_base_config' + dim_end
        else:
            CONFIG_FILE = 'marl_config' + dim_end
    else:
        CONFIG_FILE = 'config2'

    parsed = Path('learning/conf/'+CONFIG_FILE)
    hydra.initialize(str(parsed.parent),version_base='1.1')
    cfg = hydra.compose(parsed.name)

    if 'run' in args.run:
        run_rl_model(OmegaConf.to_container(cfg, resolve=True))
    if 'train' in args.run:
        train_rl_model(OmegaConf.to_container(cfg, resolve=True))
    if 'plot' in args.run:
        data_dir = collect_data(OmegaConf.to_container(cfg, resolve=True))
        action_density_plot(load_dir=data_dir)
    