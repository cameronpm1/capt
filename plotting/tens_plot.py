import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from tensorboard.backend.event_processing import event_accumulator

def find_event_files(root_dir):
    event_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if 'events.out.tfevents' in filename:
                event_files.append(os.path.join(dirpath, filename))
    return event_files

def plot_averaged_scalars(root_dirs, scalars, output_files=None,labels=None):
    colors=['tab:orange','tab:blue','tab:green']
    for i,root_dir in enumerate(root_dirs):
        print('loading', root_dir)
        event_files = find_event_files(root_dir)
        print(event_files)
        for scalar_key in scalars:
            values = []
            steps = []
            end = 40e7
            bins = 4001
            disc = np.linspace(0,end,bins)
            bin_size = end/(bins-1)
            for event_file in event_files:
                ea = event_accumulator.EventAccumulator(event_file)
                ea.Reload()
                events = ea.Scalars(scalar_key)
                for event in events:
                    if event.step > end:
                        continue
                    else:
                        steps.append(disc[int(event.step//bin_size)])
                        values.append(event.value)
            list_of_tupless = list(zip(steps,values))
            list_of_tupless

            x_label = 'Steps'
            y_label = 'Avg Episode Reward'

            df = pd.DataFrame(list_of_tupless,
                        columns=[x_label,y_label])

            line = sns.lineplot(data=df, x=x_label, y=y_label,errorbar='sd') 
            line.plot()

            #vx = [12050000,12050000,]
            
            #line.set_xticks([0, end])
            #line.set_xticklabels(['0', '40M'])
            #line.set_yticks([0, 750])
            #line.set_yticklabels([0, 750])
    plt.axvline(x=12050000, color='red', linestyle='--')
    plt.xlim((0, 30000000))
    plt.savefig('training_curve_c.png')
            


root_dir = ['/home/cameron/capt/logs/paper/baseline_tf','/home/cameron/capt/logs/paper/flagship_tf','/home/cameron/capt/logs/paper/single_tf'] #base_evader_tf'] #
scalars = ['ray/tune/env_runners/policy_reward_mean/evader'] #['ray/tune/env_runners/episode_return_mean'] #
output_files = ['ep_rew_mean_avg_plot.png', 'ep_len_mean_avg_plot.png'] 
labels = []
sns.set_style('darkgrid')
sns.set_palette('tab10')
plot_averaged_scalars(root_dir, scalars, output_files,labels=None)