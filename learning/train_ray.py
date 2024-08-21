import time

import os
import sys
import hydra
import torch
import shutil
import logging
import numpy as np
from torch import nn
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from ray.tune.logger import pretty_print
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.rllib.models import ModelCatalog
from ray.tune.logger import UnifiedLogger
from ray.tune.registry import register_env
from ray.rllib.policy.policy import Policy
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.algorithms.sac.sac_torch_policy import SACTorchPolicy
from ray.rllib.utils.metrics import (
    NUM_ENV_STEPS_SAMPLED,
)

from logger import getlogger
from learning.make_env import make_env
from envs.multi_env_wrapper import multiEnvWrapper
from envs.multi_agent_wrapper import multiAgentWrapper
from learning.custom_SAC.custom_SAC import custom_SACConfig
from custom_model_archs.sirenfcnet import SirenOutFullyConnectedNetwork


logger = getlogger(__name__)

def mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

#ModelCatalog.register_custom_model("my_torch_model", CustomTorchModel)

def train_ray(cfg: DictConfig,filedir):
    filedir = filedir
    logdir = cfg['logging']['run']['dir']
    logdir = filedir+logdir
    if not os.path.exists(logdir):
        logger.info("Save directory not found, creating path ...")
        mkdir(logdir)
    else:
        logger.info("Save directory found ...")
    print("current directory:", logdir)
    #save copy of config file
    OmegaConf.save(config=cfg, f=logdir+'/config.yaml')

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

    #wrap env for parallel single agent rl to be compatible w rllib marl
    def multi_env_maker(config):
        env = env_maker(config)  
        vec_env = multiEnvWrapper(env)
        return vec_env
    
    #wrap env for marl
    def multi_agent_env_maker(config):
        env = env_maker(config)  
        vec_env = multiAgentWrapper(env)
        return vec_env
    
    
    def rl_policy_mapping_fn(agent_id, episode, worker, **kwargs):
        '''
        ensure that parallel envs always use the 
        correct policy for single agent rl

        policies are assigned alternating
        i.e. policies: [policy0, policy1]
             workers[0,2,4,...] = policy0 
             workers[1,3,5,...] = policy1 
        '''
        idx = (worker.worker_index-1) % cfg['env']['n_policies']
        return 'policy'+str(idx)
    
    #ensure that evader and adversary agents always use the correct policy
    def marl_policy_mapping_fn(agent_id, episode, worker, **kwargs):
        '''
        ensure that evader and adversary agents always use the 
        correct policy for marl

        adversary policies assigned in alternating order same
        as single agent rl
        '''
        if 'evader' in agent_id: 
            return 'evader'
        else:
            idx = (worker.worker_index-1) % cfg['env']['n_adv_policies']
            return 'adversary'+str(idx)
        
    class policyTrainingSchedule():
        '''
        records how many batches have been used 
        '''
        def __init__(self,workers=1):
            self.workers=workers
            self.max_samples = 0

        def policy_training_schedule(self, pid, batch):
            '''
            determines when each policy should be trained
            '''
            #receives none when checking for target network update
            if batch is not None:
                self.max_samples = max(batch[pid]['unroll_id'])/len(batch.policy_batches)*self.workers
            if 'evade' in pid and self.max_samples > 9e6:
                return True
            elif 'adversary' in pid and self.max_samples < 9e6:
                return True
            else:
                return False
            
            
    policy_tracker = policyTrainingSchedule(workers=cfg['alg']['nenv'])
    
    if 'multi' in cfg['env']['scenario']:
        env_name = cfg['env']['scenario']
        register_env(env_name, multi_env_maker) #register make env function 
        #test_env for getting obs/action space
        test_env = multi_env_maker({})
        policy_list = ['policy'+str(i) for i in range(cfg['env']['n_policies'])]
        policy_mapping_fn = rl_policy_mapping_fn
        policy_training_fn = policy_list
    elif 'marl' in cfg['env']['scenario']:
        env_name = cfg['env']['scenario']
        register_env(env_name, multi_agent_env_maker) #register make env function
        #test_env for getting obs/action space
        test_env = multi_agent_env_maker({})
        policy_list = ['adversary'+str(i) for i in range(cfg['env']['n_adv_policies'])]
        policy_list.append('evader')
        policy_mapping_fn = marl_policy_mapping_fn
        policy_training_fn = policy_tracker.policy_training_schedule
    else:
        '''
        TO DO:
        add train script for non multiagent rl
        '''
        print('ERROR: single agent training not supported in train_ray.py')
        exit()
        env_name = cfg['env']['scenario']
        register_env(env_name, env_maker) #register make env function
        #test_env for getting obs/action space
        test_env = env_maker({})
        

    ModelCatalog.register_custom_model("sirenfcnet", SirenOutFullyConnectedNetwork)

    def logger_creator(config):
        return UnifiedLogger(config, logdir, loggers=None)

    batch = None

    if 'sac' in cfg['alg']['type']:
        if 'marl' in cfg['env']['scenario']:
            algo = custom_SACConfig()
            adv_policy_model_dict = {
                'custom_model': 'sirenfcnet',
                'fcnet_hiddens': cfg['alg']['pi_adv'],
            }
            adv_q_model_dict = {
                'fcnet_hiddens': cfg['alg']['vf_adv'],
            }
            evader_policy_model_dict = {
                'post_fcnet_hiddens': cfg['alg']['pi_evader'],
            }
            evader_q_model_dict = {
                'post_fcnet_hiddens': cfg['alg']['pi_evader'],
            }
            batch = cfg['alg']['batch']*(cfg['env']['n_adv_policies']+1)

            policy_info = {}
            for label in policy_list:
                if 'evader' in label:
                    policy_info[label] = (
                                    SACTorchPolicy, #policy_class
                                    test_env.get_observation_space()['evader'], #observation_space
                                    test_env.get_action_space()['evader'], #action_space
                                    {'lr':cfg['alg']['lr_evader'],
                                     'policy_model_config':evader_policy_model_dict,
                                     'q_model_config':evader_q_model_dict,
                                    }
                                )
                else:
                    policy_info[label] = (
                                    None, #policy_class
                                    test_env.get_observation_space()['adversary'], #observation_space
                                    test_env.get_action_space()['adversary'], #action_space
                                    {'lr':cfg['alg']['lr_adv'],
                                     'policy_model_config':adv_policy_model_dict,
                                     'q_model_config':adv_q_model_dict,
                                    }
                                )
        else:
            if 'div' in cfg['env']['scenario']:
                algo = custom_SACConfig()
                policy_model_dict = {
                    'custom_model': 'sirenfcnet',
                    'fcnet_hiddens': cfg['alg']['pi'],
                }
                q_model_dict = {
                    'fcnet_hiddens': cfg['alg']['vf'],
                }
            else:
                algo = SACConfig() 
                policy_model_dict = {
                    'post_fcnet_hiddens': cfg['alg']['pi'],
                }
                q_model_dict = {
                    'post_fcnet_hiddens': cfg['alg']['vf'],
                }
            batch = cfg['alg']['batch']*cfg['env']['n_policies']

            policy_info = {}
            for label in policy_list:
                policy_info[label] = (
                                None, #policy_class
                                test_env.observation_space['agent0'], #observation_space
                                test_env.action_space['agent0'], #action_space
                                {'lr':cfg['alg']['lr'],
                                 'policy_model_config':policy_model_dict,
                                 'q_model_config':q_model_dict,
                                } #config (gamma,lr,etc.)
                            )

    #initialize RL training algorithm using rllib's multi_agent settings
    algo_config = (algo
            .environment(env=env_name,
                        #env_config={'num_agents':1},
                        )
            .framework("torch")
            .env_runners(num_env_runners=cfg['alg']['nenv'], #20
                        num_envs_per_worker=cfg['alg']['cpu_envs'], #60
                        num_cpus_per_env_runner=1
                        )
            .resources(num_gpus=1)
            .multi_agent(policy_mapping_fn=policy_mapping_fn,
                            policies_to_train=policy_training_fn,
                            policies=policy_info)
            .training(gamma=cfg['alg']['gamma'], 
                        train_batch_size=batch,
                        training_intensity=cfg['alg']['train_intensity'],
                        target_entropy=cfg['alg']['target_ent'],
                        grad_clip=10,
                        grad_clip_by='norm',
                        replay_buffer_config={
                            'type': 'MultiAgentReplayBuffer', 
                            'capacity': 1000000, 
                            'replay_sequence_length': 1,
                            },
                        )
    )
    
    del test_env


    algo_build = algo_config.build(logger_creator=logger_creator)

    #set pre trained weights if training final evade or marl
    if 'evade' or 'marl' in cfg['env']['scenario']:
        pre_trained_policy = Policy.from_checkpoint(cfg['env']['evader_policy_dir'])
        pre_trained_policy_weights = pre_trained_policy.get_weights()
        if 'marl' in cfg['env']['scenario']:
            lablel = 'evader'
        else:
            lablel = 'agent0'
        pre_trained_policy_weights = {label: pre_trained_policy_weights}
        algo_build.set_weights(pre_trained_policy_weights)    

    #train 15,000 iterations
    for i in range(15000): 
        result = algo_build.train()
        if i % 500 == 0 and i != 0:
            save_dir = logdir+'/checkpoint'+str(result['timesteps_total'])
            algo_build.save(checkpoint_dir=save_dir)
            print(pretty_print(result))

'''
{'_disable_preprocessor_api': False, 
'_disable_action_flattening': False, 
'fcnet_hiddens': [256, 256], 
'fcnet_activation': 'tanh', 
'fcnet_weights_initializer': None, 
'fcnet_weights_initializer_config': None, 
'fcnet_bias_initializer': None, 
'fcnet_bias_initializer_config': None, 
'conv_filters': [[16, [3, 3], 2], [32, [2, 2], 2], [64, [1, 2], 1]], 
'conv_activation': 'relu', 
'conv_kernel_initializer': None, 
'conv_kernel_initializer_config': None, 
'conv_bias_initializer': None, 
'conv_bias_initializer_config': None, 
'conv_transpose_kernel_initializer': None, 
'conv_transpose_kernel_initializer_config': None, 
'conv_transpose_bias_initializer': None, 
'conv_transpose_bias_initializer_config': None, 
'post_fcnet_hiddens': [], 
'post_fcnet_activation': 'relu', 
'post_fcnet_weights_initializer': None, 
'post_fcnet_weights_initializer_config': None, 
'post_fcnet_bias_initializer': None, 
'post_fcnet_bias_initializer_config': None,
'free_log_std': False, 
'no_final_linear': False, 
'vf_share_layers': True, 
'use_lstm': False,
'max_seq_len': 20, 
'lstm_cell_size': 256, 
'lstm_use_prev_action': False,
'lstm_use_prev_reward': False,
'lstm_weights_initializer': None,
'lstm_weights_initializer_config': None,
'lstm_bias_initializer': None, 
'lstm_bias_initializer_config': None, 
'_time_major': False, 
'use_attention': False, 
'attention_num_transformer_units': 1, 
'attention_dim': 64, 
'attention_num_heads': 1, 
'attention_head_dim': 32, 
'attention_memory_inference': 50, 
'attention_memory_training': 50, 
'attention_position_wise_mlp_dim': 32, 
'attention_init_gru_gate_bias': 2.0, 
'attention_use_n_prev_actions': 0, 
'attention_use_n_prev_rewards': 0, 
'framestack': True, 
'dim': 84, 
'grayscale': False, 
'zero_mean': True, 
'custom_model': None, 
'custom_model_config': {}, 
'custom_action_dist': None, 
'custom_preprocessor': None, 
'encoder_latent_dim': None, 
'always_check_shapes': False, 
'lstm_use_prev_action_reward': -1, 
'_use_default_native_models': -1}
'''

#'conv_filters': [[32, [3, 3], 4], [64, [2, 2], 2], [64, [1, 1], 1]],
#'_disable_preprocessor_api': True,