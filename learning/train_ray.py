import time

import os
import sys
import hydra
import torch
import shutil
import logging
import numpy as np
from torch import nn
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from ray.tune.logger import pretty_print
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.rllib.models import ModelCatalog
from ray.tune.logger import UnifiedLogger
from ray.tune.registry import register_env
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

from logger import getlogger
from learning.make_env import make_env
from envs.evade_pursuit_env import evadePursuitEnv
from envs.multi_env_wrapper import multiEnvWrapper

from complex_input_net import ComplexInputNetwork


logger = getlogger(__name__)

def mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

class CustomTorchModel(TorchModelV2):
    def __init__(
        self, 
        obs_space, 
        action_space, 
        num_outputs, 
        model_config, 
        name,
    ) -> None:

        n_input_channels = obs_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())




    def forward(self, input_dict, state, seq_lens): ...
    def value_function(self): ...

#ModelCatalog.register_custom_model("my_torch_model", CustomTorchModel)


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

    ModelCatalog.register_custom_model("my_tf_model", ComplexInputNetwork)

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
                .training(gamma=cfg['alg']['gamma'], 
                            train_batch_size=cfg['alg']['batch'],
                            training_intensity=cfg['alg']['train_intensity'],
                            target_entropy=cfg['alg']['target_ent'],
                            replay_buffer_config={
                                'type': 'MultiAgentReplayBuffer', 
                                'capacity': 1000000, 
                                'replay_sequence_length': 1,
                                },
                            policy_model_config={
                                #'custom_model': 'my_tf_model',
                                'conv_filters': [[32, [3, 3], 4], [64, [2, 2], 2], [64, [1, 1], 1]],
                                #'post_fcnet_hiddens': cfg['alg']['pi'],
                                'post_fcnet_hiddens': cfg['alg']['pi'],
                                #'_disable_preprocessor_api': True,
                            },
                            q_model_config={
                                'conv_filters': [[32, [3, 3], 4], [64, [2, 2], 2], [64, [1, 1], 1]],
                                #'post_fcnet_hiddens': cfg['alg']['vf'],
                                'post_fcnet_hiddens': cfg['alg']['vf'],
                                #'_disable_preprocessor_api': True,
                            },
                            )
                #.rollout(batch_mode='truncated_episods',
                #            rollout_fragment_length=256,)
        )
    
    del test_env

    algo_build = algo_config.build(logger_creator=logger_creator)
    #result = algo_build.train()
    #print(pretty_print(result))
    #model = algo_build.get_policy().model
    #model_out = model({"obs": np.array([[0.1, 0.2, 0.3, 0.4]])})
    #model.base_model.summary()
    #t0 = time.time()


    for i in range(12000):
        result = algo_build.train()
        if i % 400 == 0 and i != 0:
            save_dir = logdir+'/checkpoint'+str(result['timesteps_total'])
            algo_build.save(checkpoint_dir=save_dir)
            print(pretty_print(result))

    #td = time.time()-t0
    #print(td)

    #checkpoint_dir = algo_build.save(checkpoint_dir=logdir).checkpoint.path


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