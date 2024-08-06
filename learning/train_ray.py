import time

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
from envs.multi_env_wrapper import multiEnvWrapper


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
    seed_offset = 0

    #make env function 
    def env_maker(config):
        str(config) #needed to call worker and vector _index (odd bug w rllib)
        env = make_env(filedir,cfg)
        #if config has parameters, ensure envs have different seeds
        seed = cfg['seed']
        if len(config.keys()) > 0:
            seed += ((100*config.worker_index) + config.vector_index)
        env.unwrapped.seed(seed)
        return env

    def multi_env_maker(config):
        #[env_maker({},seed=(cfg['seed'] + (i*100))) for i in range(#cfg['env']['nenvs'])]
        env = env_maker(config)  
        vec_env = multiEnvWrapper(env)
        return vec_env
    
    if 'multi' in cfg['env']['scenario']:
        env_name = cfg['env']['scenario']
        register_env(env_name, multi_env_maker) #register make env function 
        #test_env for getting obs/action space
        test_env = multi_env_maker({})
        policy_list = ['policy'+str(i) for i in range(cfg['env']['n_policies'])]
    elif 'control' in cfg['env']['scenario']:
        env_name = cfg['env']['scenario']
        register_env(env_name, env_maker) #register make env function
        #test_env for getting obs/action space
        test_env = env_maker({})
        

    #ensure that evader and adversary agents always use the correct policy
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        '''
        each agent is assigned to a single worker
        '''
        idx = (worker.worker_index-1) % cfg['env']['n_policies']
        return 'policy'+str(idx)

    def logger_creator(config):
        return UnifiedLogger(config, logdir, loggers=None)

    if 'sac' in cfg['alg']['type']:
        algo = SACConfig()

    if 'multi' in cfg['env']['scenario']:
        #initialize MARL training algorithm

        policy_info = {}
        for label in policy_list:
            policy_info[label] = (
                                    None, #policy_class
                                    test_env.observation_space['agent0'], #observation_space
                                    test_env.action_space['agent0'], #action_space
                                    {} #config (gamma,lr,etc.)
                                )
        algo_config = (algo
                .environment(env=env_name,
                            env_config={'num_agents':1},)
                .framework("torch")
                .env_runners(num_env_runners=20, #20
                            num_envs_per_worker=60, #60
                            num_cpus_per_env_runner=1
                            )
                .resources(num_gpus=1)
                .multi_agent(policy_mapping_fn=policy_mapping_fn,
                            policies=policy_info)
                .training(gamma=0.99, 
                            train_batch_size=256,
                            optimization_config={
                                'actor_learning_rate': cfg['alg']['lr'],
                                'critic_learning_rate': cfg['alg']['lr'],
                                'entropy_learning_rate': 0
                                },
                            replay_buffer_config={
                                'type': 'MultiAgentReplayBuffer', 
                                'capacity': 1000000, 
                                'replay_sequence_length': 1,
                                })
        )
    
    del test_env

    algo_build = algo_config.build(logger_creator=logger_creator)
    #result = algo_build.train()
    #print(pretty_print(result))
    #model = algo_build.get_policy().model
    #model_out = model({"obs": np.array([[0.1, 0.2, 0.3, 0.4]])})
    #model.base_model.summary()
    #t0 = time.time()


    for i in range(10000):
        result = algo_build.train()
        if i % 400 == 0 and i != 0:
            save_dir = logdir+'/checkpoint'+str(result['timesteps_total'])
            algo_build.save(checkpoint_dir=save_dir)
            print(pretty_print(result))

    #td = time.time()-t0
    #print(td)

    #checkpoint_dir = algo_build.save(checkpoint_dir=logdir).checkpoint.path




