import os
import sys
import hydra
import torch
import shutil
import logging
import gymnasium
import numpy as np
import torch as th
from torch import nn
from typing import Dict
from gymnasium import spaces
from omegaconf import DictConfig
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from hydra.core.hydra_config import HydraConfig
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space


from logger import getlogger
from learning.make_env import make_env
from learning.subproc_vec_env_no_daemon import SubprocVecEnvNoDaemon

logger = getlogger(__name__)

def mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

class CustomCNN(BaseFeaturesExtractor):
    """
    same as NatureCNN from torch_layers with custom kernel
    """

    def __init__(
        self,
        observation_space: gymnasium.Space,
        features_dim: int = 512,
        normalized_image: bool = False,
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "NatureCNN must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space, check_channels=False, normalized_image=normalized_image), (
            "You should use NatureCNN "
            f"only with images not with {observation_space}\n"
            "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
            "If you are using a custom environment,\n"
            "please check it using our env checker:\n"
            "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html.\n"
            "If you are using `VecNormalize` or already normalized channel-first images "
            "you should pass `normalize_images=False`: \n"
            "https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html"
        )
        n_input_channels = observation_space.shape[0]
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

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

class CustomCombinedExtractor(BaseFeaturesExtractor):
    """
    Combined features extractor for Dict observation spaces.
    Builds a features extractor for each key of the space. Input from each space
    is fed through a separate submodule (CNN or MLP, depending on input shape),
    the output features are concatenated and fed through additional MLP network ("combined").

    :param observation_space:
    :param cnn_output_dim: Number of features to output from each CNN submodule(s). Defaults to
        256 to avoid exploding network sizes.
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        cnn_output_dim: int = 256,
        normalized_image: bool = False,
    ) -> None:
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super().__init__(observation_space, features_dim=1)

        extractors: Dict[str, nn.Module] = {}

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace, normalized_image=normalized_image):
                extractors[key] = CustomCNN(subspace, features_dim=cnn_output_dim, normalized_image=normalized_image)
                total_concat_size += cnn_output_dim
            else:
                # The observation key is a vector, flatten it if needed
                extractors[key] = nn.Flatten()
                total_concat_size += get_flattened_obs_dim(subspace)

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations: TensorDict) -> th.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return th.cat(encoded_tensor_list, dim=1)

def train(cfg: DictConfig,filedir):
    filedir = filedir
    set_random_seed(cfg['seed'])
    logdir = cfg['logging']['run']['dir']
    logdir = filedir+logdir
    if not os.path.exists(logdir):
        logger.info("Safe directory not found, creating path ...")
        mkdir(logdir)
    else:
        logger.info("Save directory found ...")
    print("current directory:", logdir)
    #logging.basicConfig(filename=logdir+'\log.log') #set up logger file

    # make parallel envs
    def env_fn(seed):
        env = make_env(filedir,cfg)
        env.unwrapped.seed(seed)
        return env

    class EnvMaker:
        def __init__(self, seed):
            self.seed = seed

        def __call__(self):
            return env_fn(self.seed)


    def make_vec_env(nenvs, seed):
        envs = VecMonitor(
            SubprocVecEnvNoDaemon([EnvMaker(seed + i) for i in range(nenvs)])
        )
        return envs

    logger.info("Preparing to initialize parallel environments ...")
    env = make_vec_env(cfg["alg"]["nenv"], cfg["seed"])
    logger.info("Parallel environments initialized ...")
    # define policy network size
    
    

    # alg kw for sac or td3
    if cfg["alg"]["type"] == "ppo":
        alg = PPO
        policy_kwargs = dict(net_arch=dict(pi=cfg["alg"]["pi"], vf=cfg["alg"]["vf"]))
        alg_kwargs = {}

    if cfg["alg"]["type"] == "sac":
        alg = SAC
        if 'Image' in cfg['env']['scenario']:
            policy_kwargs = dict(
                net_arch=dict(pi=cfg["alg"]["pi"], qf=cfg["alg"]["vf"]), 
                features_extractor_class=CustomCombinedExtractor,
                features_extractor_kwargs=dict(normalized_image=True))
        else:
            policy_kwargs = dict(
                net_arch=dict(pi=cfg["alg"]["pi"], qf=cfg["alg"]["vf"]))
        alg_kwargs = {}

    if 'Image' in cfg['env']['scenario']:
        policy = 'MultiInputPolicy'
    else:
        policy = 'MlpPolicy'

    logger.info("Initializing model ...")
    # define model
    model = alg(
            policy, 
            env, 
            verbose=1,
            tensorboard_log=logdir,
            learning_rate=cfg["alg"]["lr"], 
            batch_size=cfg["alg"]["batch_size"],
            gamma=cfg["alg"]["gamma"],
            policy_kwargs=policy_kwargs,
            seed=cfg["seed"],
            device=cfg["alg"]["device"],
            **alg_kwargs,
            )

    checkpoint_callback = CheckpointCallback(
            save_freq=100000,
            save_path=logdir,
            name_prefix="midtrain_model",
            save_replay_buffer=False,
            save_vecnormalize=False,
            )


    logger.info("Beginning training ...")
    # train
    model.learn(
            total_timesteps=cfg["alg"]["total_timesteps"],
            progress_bar=True,
            callback=checkpoint_callback,
            )


    # end
    env.close()
    logger.info("Saving model ...")
    model.save(os.path.join(logdir, "model"))

def retrain(cfg: DictConfig,filedir,modeldir):
    filedir = filedir
    set_random_seed(cfg['seed'])
    logdir = cfg['logging']['run']['dir']
    logdir = filedir+logdir
    if not os.path.exists(logdir):
        logger.info("Safe directory not found, creating path ...")
        mkdir(logdir)
    else:
        logger.info("Save directory found ...")
    print("current directory:", logdir)
    #logging.basicConfig(filename=logdir+'\log.log') #set up logger file

    # make parallel envs
    def env_fn(seed):
        env = make_env(filedir,cfg)
        env.unwrapped.seed(seed)
        return env

    class EnvMaker:
        def __init__(self, seed):
            self.seed = seed

        def __call__(self):
            return env_fn(self.seed)


    def make_vec_env(nenvs, seed):
        envs = VecMonitor(
            SubprocVecEnvNoDaemon([EnvMaker(seed + i) for i in range(nenvs)])
        )
        return envs

    logger.info("Preparing to initialize parallel environments ...")
    env = make_vec_env(cfg["alg"]["nenv"], cfg["seed"])
    logger.info("Parallel environments initialized ...")
    # define policy network size
    

    if cfg["alg"]["type"] == "ppo":
        alg = PPO
        logger.info("Loading model ...")
        model = PPO.load(modeldir)
    elif cfg["alg"]["type"] == "sac":
        alg = SAC
        logger.info("Loading model ...")
        model = SAC.load(modeldir)


    model.tensorboard_log = logdir
    logger.info("Loading parallel environments ...")
    # load env
    model.set_env(env)

    logger.info("Loading model ...")
    # load model
    model = PPO.load(modeldir)
    model.tensorboard_log = logdir

    logger.info("Loading parallel environments ...")
    # load env
    model.set_env(env)

    checkpoint_callback = CheckpointCallback(
            save_freq=50000,
            save_path=logdir,
            name_prefix="retrain_model",
            save_replay_buffer=True,
            save_vecnormalize=True,
            )


    logger.info("Beginning training ...")
    # train
    model.learn(
            total_timesteps=cfg["alg"]["total_timesteps"],
            progress_bar=True,
            callback=checkpoint_callback,
            )


    # end
    env.close()
    logger.info("Saving model ...")
    model.save(os.path.join(logdir, "retrained_model"))



if __name__ == "__main__":
    torch.set_num_threads(9)
    DIRECTORY = os.getcwd()
    train()