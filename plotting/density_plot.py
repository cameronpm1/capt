import os
import time
import torch
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from omegaconf import DictConfig

from ray.rllib.policy.policy import Policy
from ray.rllib.algorithms.sac.sac_torch_policy import _get_dist_class

from learning.make_env import make_env
from sim_prompters.twod_marl_prompter import twodMARLPrompter
from sim_prompters.threed_marl_prompter import threedMARLPrompter


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
    iter = 4

    evader = None
    models = []
    policies = []
    envs = []
    action_dist_data = []

    model_dirs = os.listdir(master_dir)

    dummy_prompter1 = dummy_prompter()

    for model_dir in model_dirs:
        if 'adversary' in model_dir:
            policy = Policy.from_checkpoint(master_dir+'/'+model_dir)
            policies.append(policy)
            model = policy.model
            models.append(model)

            filedir = filedir
            env = make_env(filedir,cfg)
            env.unwrapped.seed(seed=cfg['seed'])
            env.unwrapped.prompter=dummy_prompter1
            envs.append(env)

            action_dist_data.append(np.array([]))        
        elif 'evader' in model_dir:
            policy = Policy.from_checkpoint(master_dir+'/'+model_dir)
            evader = policy
    print(policies)

    #use same prompter for all envs
    if cfg['env']['dim'] == 2:
            prompter = twodMARLPrompter()
    if cfg['env']['dim'] == 3:
            prompter = threedMARLPrompter()
    prompter.seed(seed=cfg['seed'])
    prompter.set_num_adv(len(models))

    end_timestep = cfg['env']['max_timestep']

    kl_div = []

    for i in range(iter):
        print('running simulation',i)
        observations = []
        timesteps = []
        dones = []
        dummy_prompter1.set_prompt(prompter.prompt())

        for j,env in enumerate(envs):
            observations.append([])
            dones.append([])
            obs, _ = env.reset()
            observations[j].append(obs)
            timesteps.append(0)
            dones.append(False)

        for t in range(end_timestep):
            for j,env in enumerate(envs):
                if not dones[j]:
                    #compute next action and take forward step
                    action_dict = {}
                    action_dict['evader'],_,_ = evader.compute_single_action(observations[j][-1]['evader'])
                    action_dict['adversary0'],_,_ = policies[j].compute_single_action(observations[j][-1]['adversary0'])
                    action_dict['adversary1'],_,_ = policies[j].compute_single_action(observations[j][-1]['adversary1'])
                    #action_dict['adversary1'],_,_ = policies[i%cfg['env']['n_policies']].compute_single_action(observations[j][-1]['adversary1'])
                    obs,rewards,terminated,truncated,_ = env.step(action_dict)
                    observations[j].append(obs)
                    if terminated['__all__'] or truncated['__all__']:
                        dones[j] = True
                        print(t)
                    else:
                        dones[j] = False

        cc_observations = None

        obs_len = 0
        for k,obs_set in enumerate(observations):
             obs_len += len(obs_set)

             if k == 0:
                  cc_observations = np.array(obs_set)
             else:
                  cc_observations = np.concatenate((cc_observations,np.array(obs_set)))

        observations = cc_observations

        dists = []
        last_dist = None

        for j,env in enumerate(envs):
            #compute action dist outputs
            obs = np.array([obs['adversary0'] for obs in observations])
            #obs = np.concatenate((obs,np.array([obs['adversary1'] for obs in observations])))
            model_out, _ = models[j]({'obs':obs.squeeze()})
            actions_input, _ = models[j].get_action_model_outputs(torch.from_numpy(model_out).cuda())
            action_dist_class = _get_dist_class(policies[j], policies[j].config, policies[j].action_space)
            action_dist = action_dist_class(actions_input, models[j])
            if i == 0:
                 dists.append(action_dist)
            if j == 0:
                 last_dist = action_dist
            else:
                 kl_div.append(torch.mean(last_dist.kl(action_dist)).item())
                 kl_div.append(torch.mean(action_dist.kl(last_dist)).item())
            #divergent_action_input = [mean1,mean2,log_std1,log_std2]
            episode_dist_data = np.concatenate((action_dist.mean.cpu().detach().numpy(),action_dist.std.cpu().detach().numpy()),axis=1)
            #action_dist_data[j].append((action_dist.mean.cpu().detach().numpy(),action_dist.std.cpu().detach().numpy()))
            if len(action_dist_data[j]) > 0 :
                action_dist_data[j] = np.concatenate((action_dist_data[j],episode_dist_data),axis=0)
            else:
                action_dist_data[j] = episode_dist_data
        '''
        if i == 0:
            for a in range(2):
                for b in range(2):
                    if a != b:
                        print(torch.mean(dists[a].kl(dists[b])))
        '''
    print(np.average(kl_div))

    file_dirs = os.listdir(master_dir)
    for file in file_dirs:
         if 'data' in file:
              os.remove(master_dir+'/'+file)

    for i,array in enumerate(action_dist_data):
        np.save(master_dir+'/'+'data'+str(i)+'.npy',array)

# explicit function to normalize array
def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)    
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr

def action_density_plot(
        load_dir: str,
):
    files = os.listdir(load_dir)
    data = []

    for file in files:
         if 'data' in file:
              data.append(np.load(load_dir+'/'+file))

    for i in range(len(data)):
         print('plotting policy'+str(i)+' data')
         grid_size = 50
         iter = 2/grid_size
         grid = np.zeros((grid_size,grid_size))
         if data[i][:,].max() > 1 or data[i][:,].min() < -1:
              data1 = normalize(data[i][:,0],-1,1)
              data2 = normalize(data[i][:,3],-1,1)
              normalize_on = True
         else:
              normalize_on = False
         for j,data_point in enumerate(data[i]):
            if normalize_on:
                 idx1, idx2 = (data1[j] + 1)//iter , (data2[j] + 1)//iter
            else:
                idx1, idx2 = (data_point[0] + 1)//iter , (data_point[1] + 1)//iter
            if idx1 > grid_size - 1:
                 idx1 -= grid_size - 1
            if idx2 > grid_size - 1:
                 idx2 = grid_size - 1  
            grid[int(idx1),int(idx2)] += 1
         grid /= len(data[i])
        #compute prob contour lines
         n = 100
         t = np.linspace(0, grid.max(), n)
         integral = ((grid >= t[:, None, None]) * grid).sum(axis=(1,2))
         f = interpolate.interp1d(integral, t)
         t_contours = f(np.array([0.9,0.7,0.5,0.3]))
         plt.subplot(1, len(data), i+1)
         plt.imshow(grid.T, origin='lower', extent=[-1,1,-1,1], cmap="gray")
         plt.contour(grid.T, t_contours, extent=[-1,1,-1,1])

        #mean1 = data[i][:,0]
        #mean2 = data[i][:,1]
        #
        #plt.scatter(mean1,mean2,s=1)
    plt.savefig(load_dir+'/'+'density.png')


if __name__ == "__main__":
    action_density_plot(load_dir='/home/cameron/magpie_rl/logs/test_div_policies')