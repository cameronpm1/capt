import os
import sys
import hydra
import torch
import shutil
import logging
import numpy as np
from omegaconf import DictConfig
from ray.tune.logger import pretty_print
from hydra.core.hydra_config import HydraConfig
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.tune.logger import UnifiedLogger
from ray.tune.registry import register_env
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.algorithms.ppo import PPOConfig

from logger import getlogger
from learning.make_env import make_env
from envs.evade_pursuit_env import evadePursuitEnv


logger = getlogger(__name__)


def mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)


def train_ray(cfg: DictConfig,filedir):
    filedir = filedir
    logdir = cfg['logging']['run']['dir']
    logdir = filedir+logdir
    if not os.path.exists(logdir):
        logger.info("Safe directory not found, creating path ...")
        mkdir(logdir)
    else:
        logger.info("Save directory found ...")
    print("current directory:", logdir)
    #logging.basicConfig(filename=logdir+'\log.log') #set up logger file

    #make env function 
    def env_maker(config):
        env = make_env(filedir,cfg)
        env.seed(cfg['seed'])
        return env
    
    if 'multi' in cfg['env']['scenario']:
        env_name = 'evadePursuitEnv'
        register_env(env_name, env_maker) #register make env function 
    if 'control' in cfg['env']['scenario']:
        env_name = 'controlEnv'
        register_env(env_name, env_maker) #register make env function

    #ensure that evader and adversary agents always use the correct policy
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        if agent_id.startswith('evader'):
            return 'evader'
        if agent_id.startswith('adversary'):
            return 'adversary'
        print('Error: unknown agent id')
        exit()

    def logger_creator(config):
        return UnifiedLogger(config, logdir, loggers=None)

    if 'sac' in cfg['alg']['type']:
        algo = SACConfig()

    #test_env for getting obs/action space
    test_env = env_maker({})

    if 'multi' in cfg['env']['scenario']:
        #initialize MARL training algorithm
        algo_config = (algo
                .environment(env=env_name,
                            env_config={'num_agents':2},)
                .framework("torch")
                .env_runners(num_env_runners=20,
                            num_envs_per_worker=20,
                            num_cpus_per_env_runner=1
                            )
                .resources(num_gpus=1)
                .multi_agent(policy_mapping_fn=policy_mapping_fn,
                            policies={
                                'evader':(
                                    None, #policy_class
                                    test_env.observation_space['evader'], #observation_space
                                    test_env.action_space['evader'], #action_space
                                    {} #config (gamma,lr,etc.)
                                ),
                                'adversary':(
                                    None, #policy_class
                                    test_env.observation_space['adversary'], #observation_space
                                    test_env.action_space['adversary'], #action_space
                                    {} #config (gamma,lr,etc.)
                                ),
                            })
                .training(gamma=0.99, 
                            lr=0.0001,
                            train_batch_size=256,
                            replay_buffer_config={
                                'type': 'MultiAgentReplayBuffer', 
                                'capacity': 1000000, 
                                'replay_sequence_length': 1,
                                })
        )
    else:
        #initialize single training algorithm
        algo_config = (algo
                .environment(env=env_name)
                .framework('torch')
                .env_runners(num_env_runners=1,
                            num_envs_per_worker=1,
                            num_cpus_per_env_runner=1
                            )
                .resources(num_gpus=1)
                .training(gamma=0.99, 
                            lr=0.0001,
                            train_batch_size=256,
                            replay_buffer_config={
                                'capacity': 1000000, 
                                })
        )
    
    del test_env

    algo_build = algo_config.build(logger_creator=logger_creator)
    result = algo_build.train()
    #print(pretty_print(result))
    #model = algo_build.get_policy().model
    #model_out = model({"obs": np.array([[0.1, 0.2, 0.3, 0.4]])})
    #model.base_model.summary()

    '''
    for i in range(15000):
        result = algo_build.train()
        print(pretty_print(result))

        if i % 1000 == 0 and i != 1:
            checkpoint_dir = algo_build.save(checkpoint_dir=logdir).checkpoint.path
            print(f"Checkpoint saved in directory {checkpoint_dir}")
    '''



