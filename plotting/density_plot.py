import torch
import numpy as np
import matplotlib.pyplot as plt

from ray.rllib.policy.policy import Policy
from ray.rllib.algorithms.sac.sac_torch_policy import _get_dist_class


def rllib_action_density_plot(
        policy_dir: str,
):

    data = {"obs": np.array([[0.1, 0.2, 0.3, 0.4]])}

    policy = Policy.from_checkpoint(policy_dir)
    model = policy.model

    model_out, _ = model(data) #, [], None)
    divergent_actions_input, _ = model.get_action_model_outputs(torch.from_numpy(model_out).cuda())
    #divergent_action_input = [mean1,mean2,log_std1,log_std2]
    action_dist_class = _get_dist_class(policy, policy.config, policy.action_space)
    action_dist = action_dist_class(divergent_actions_input, model)
    print(action_dist.mean,action_dist.std)

if __name__ == "__main__":
    rllib_action_density_plot(policy_dir='/home/cameron/magpie_rl/logs/adv_nogoal4/2024-08-14/checkpoint13202400/policies/policy0')