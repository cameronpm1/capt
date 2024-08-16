import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig

from ray.rllib.policy.policy import Policy
from ray.rllib.algorithms.sac.sac_torch_policy import _get_dist_class

from learning.make_env import make_env
from sim_prompters.one_v_one_prompter import oneVOnePrompter
from sim_prompters.twod_one_v_one_prompter import twodOneVOnePrompter


#ModelCatalog.register_custom_model("my_torch_model", CustomTorchModel)

class dummy_prompter():
    def __init__(self):
         self.current_prompt = None

    def prompt(self):
         return self.current_prompt
    
    def set_prompt(self,prompt):
         self.current_prompt = prompt

def collect_action_dist_data(
        cfg : DictConfig, 
        filedir: str,
        master_dir: str,
):
    iter = 100

    models = []
    policies = []
    envs = []
    action_dist_data = []

    model_dirs = os.listdir(master_dir)

    dummy_prompter1 = dummy_prompter()

    for model_dir in model_dirs:
        policy = Policy.from_checkpoint(master_dir+'/'+model_dir)
        policies.append(policy)
        model = policy.model
        models.append(model)

        filedir = filedir
        env = make_env(filedir,cfg)
        env.unwrapped.seed(seed=cfg['seed'])
        env.unwrapped.prompter=dummy_prompter1
        envs.append(env)

        action_dist_data.append([])        

    #use same prompter for all envs
    if cfg['env']['dim'] == 2:
            prompter = twodOneVOnePrompter()
    if cfg['env']['dim'] == 3:
            prompter = oneVOnePrompter()
    prompter.seed(seed=cfg['seed'])

    end_timestep = cfg['env']['max_timestep']

    for i in range(iter):
        observations = []
        timesteps = []
        dones = []
        dummy_prompter1.set_prompt(prompter.prompt())

        for env in envs:
            obs, _ = env.reset()
            observations.append(obs)
            timesteps.append(0)
            dones.append(False)

        for t in range(end_timestep):
            for j,env in enumerate(envs):
                if not dones[j]:
                    #compute action dist outputs
                    model_out, _ = models[j]({'obs':np.array([observations[j]])})
                    actions_input, _ = model.get_action_model_outputs(torch.from_numpy(model_out).cuda())
                    action_dist_class = _get_dist_class(policies[j], policies[j].config, policies[j].action_space)
                    action_dist = action_dist_class(actions_input, models[j])
                    #divergent_action_input = [mean1,mean2,log_std1,log_std2]
                    action_dist_data[j].append((action_dist.mean.cpu().detach().numpy(),action_dist.std.cpu().detach().numpy()))

                    #compute next action and take forward step
                    action,_,_ = policies[j].compute_single_action(observations[j])
                    obs,rewards,terminated,truncated,_ = env.step(action)
                    observations[j] = obs
                    if terminated or truncated:
                        dones[j] = True
                    else:
                        dones[j] = False

    #filedir = filedir
    #logdir = master_dir + '/data.npy'
    #logdir = filedir+logdir
    #with open(logdir, 'wb') as f:
    for i,array in enumerate(action_dist_data):
        np.save(master_dir+'/'+'data'+str(i)+'.npy',array)


def action_density_plot(
        load_dir: str,
):

    files = model_dirs = os.listdir(load_dir)

    data = []

    for file in files:
         if 'data' in file:
              data.append(np.load(load_dir+'/'+file))
    
    mean11 = data[0].squeeze()[:,0,0]
    mean12 = data[0].squeeze()[:,0,1]
    mean21 = data[1].squeeze()[:,0,0]
    mean22 = data[1].squeeze()[:,0,1]
    plt.subplot(1, 2, 1)
    plt.scatter(mean11,mean12,s=1)
    plt.subplot(1, 2, 2)
    plt.scatter(mean21,mean22,s=1)

    plt.savefig('test.png')


if __name__ == "__main__":
    action_density_plot(load_dir='/home/cameron/magpie_rl/logs/test_div_policies')