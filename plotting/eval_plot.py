import os
import time
import torch
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from omegaconf import DictConfig,open_dict

from learning.make_env import make_env
from learning.run_space_sim_ray import runSpaceSimRay

def eval_plot(
        cfg : DictConfig, 
        directory: str,
        master_dir: str,
):
    
    checkpoint_dirs = os.listdir(master_dir)
    checkpoint_map = {}
    start=14000000

    #save directories with checkpoint
    for dir in checkpoint_dirs:
        if 'checkpoint' in dir:
            if int(dir[10:]) > start:
                checkpoint_map[dir] = int(dir[10:])
    #sort checkpoint directories
    sorted_dirs = sorted(checkpoint_map.keys(), key=checkpoint_map.get)

    first = []
    second = []

    for dir in sorted_dirs:
        print(master_dir+'/'+dir)
        cfg['env']['evader_policy_dir'] = master_dir+'/'+dir+'/policies/evader'
        res = runSpaceSimRay(cfg,directory,modeldir=None,render=False,verbose=False)
        first.append(res[0])
        second.append(res[1])

    x = np.linspace(0,int(sorted_dirs[-1][10:]),len(sorted_dirs))

    plt.plot(x,first)
    plt.plot(x,second)
    plt.savefig(master_dir+'/'+'eval.png')

    