"""
PyTorch policy class used for SAC.
"""
import torch
import logging
from torch import nn
import gymnasium as gym
from typing import Dict, List, Optional, Tuple, Type, Union

import ray
import ray.experimental.tf_utils
from ray.rllib.policy.policy import Policy
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.typing import TensorType
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.utils.spaces.simplex import Simplex
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import AlgorithmConfigDict
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.policy.torch_mixins import TargetNetworkMixin
from ray.rllib.policy.policy_template import build_policy_class
from ray.rllib.algorithms.dqn.dqn_tf_policy import PRIO_WEIGHTS
from ray.rllib.algorithms.sac.sac_tf_policy import build_sac_model
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.utils.torch_utils import (
    apply_grad_clipping,
    concat_multi_gpu_td_errors,
    huber_loss,
)
from ray.rllib.algorithms.sac.sac_tf_policy import (
    postprocess_trajectory,
    validate_spaces,
)
from ray.rllib.models.torch.torch_action_dist import (
    TorchCategorical,
    TorchDistributionWrapper,
    TorchDirichlet,
    TorchSquashedGaussian,
    TorchDiagGaussian,
    TorchBeta,
)
from ray.rllib.algorithms.sac.sac_torch_policy import(
    stats,
    optimizer_fn,
    setup_late_mixins,
    ComputeTDErrorMixin,
    action_distribution_fn,
    build_sac_model_and_action_dist,
)

F = nn.functional

logger = logging.getLogger(__name__)

def _get_dist_class(
    policy: Policy, config: AlgorithmConfigDict, action_space: gym.spaces.Space
) -> Type[TorchDistributionWrapper]:
    """Helper function to return a dist class based on config and action space.

    Args:
        policy: The policy for which to return the action
            dist class.
        config: The Algorithm's config dict.
        action_space (gym.spaces.Space): The action space used.

    Returns:
        Type[TFActionDistribution]: A TF distribution class.
    """
    if hasattr(policy, "dist_class") and policy.dist_class is not None:
        return policy.dist_class
    elif config["model"].get("custom_action_dist"):
        action_dist_class, _ = ModelCatalog.get_action_dist(
            action_space, config["model"], framework="torch"
        )
        return action_dist_class
    elif isinstance(action_space, gym.spaces.Discrete):
        return TorchCategorical
    elif isinstance(action_space, Simplex):
        return TorchDirichlet
    else:
        assert isinstance(action_space, gym.spaces.Box)
        if config["normalize_actions"]:
            return (
                TorchSquashedGaussianKL
                if not config["_use_beta_distribution"]
                else TorchBeta
            )
        else:
            return TorchDiagGaussian

class TorchSquashedGaussianKL(TorchSquashedGaussian):

    def __init__(
        self,
        inputs: Union[float, torch.Tensor],
        model: Optional[Union[float, torch.Tensor]] = 1.0,
        low: float = -1.0,
        high: float = 1.0,
    ):
        super().__init__(
            inputs,
            model,
            low=low,
            high=high
        )

    def kl(self, other: ActionDistribution) -> TensorType:
        return torch.distributions.kl.kl_divergence(self.dist, other.dist)
    
def build_sac_model_and_action_dist(
    policy: Policy,
    obs_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
    config: AlgorithmConfigDict,
) -> Tuple[ModelV2, Type[TorchDistributionWrapper]]:
    """Constructs the necessary ModelV2 and action dist class for the Policy.

    Args:
        policy: The TFPolicy that will use the models.
        obs_space (gym.spaces.Space): The observation space.
        action_space (gym.spaces.Space): The action space.
        config: The SACConfig object.

    Returns:
        ModelV2: The ModelV2 to be used by the Policy. Note: An additional
            target model will be created in this function and assigned to
            `policy.target_model`.
    """
    model = build_sac_model(policy, obs_space, action_space, config)
    action_dist_class = _get_dist_class(policy, config, action_space)
    return model, action_dist_class

def custom_actor_critic_loss(
    policy: Policy,
    model: ModelV2,
    dist_class: Type[TorchDistributionWrapper],
    train_batch: SampleBatch,
) -> Union[TensorType, List[TensorType]]:
    """Constructs the loss for the Soft Actor Critic.
    
    Args:
        policy: The Policy to calculate the loss for.
        model (ModelV2): The Model to calculate the loss for.
        dist_class (Type[TorchDistributionWrapper]: The action distr. class.
        train_batch: The training data.

    Returns:
        Union[TensorType, List[TensorType]]: A single loss tensor or a list
            of loss tensors.
    """

    # Look up the target model (tower) using the model tower.
    target_model = policy.target_models[model]

    # Should be True only for debugging purposes (e.g. test cases)!
    deterministic = policy.config["_deterministic_loss"]

    model_out_t, _ = model(
        SampleBatch(obs=train_batch[SampleBatch.CUR_OBS], _is_training=True), [], None
    )
    
    model_out_tp1, _ = model(
        SampleBatch(obs=train_batch[SampleBatch.NEXT_OBS], _is_training=True), [], None
    )

    target_model_out_tp1, _ = target_model(
        SampleBatch(obs=train_batch[SampleBatch.NEXT_OBS], _is_training=True), [], None
    )

    alpha = torch.exp(model.log_alpha)

    # Discrete case.
    if model.discrete:
        # Get all action probs directly from pi and form their logp.
        action_dist_inputs_t, _ = model.get_action_model_outputs(model_out_t)
        log_pis_t = F.log_softmax(action_dist_inputs_t, dim=-1)
        policy_t = torch.exp(log_pis_t)
        action_dist_inputs_tp1, _ = model.get_action_model_outputs(model_out_tp1)
        log_pis_tp1 = F.log_softmax(action_dist_inputs_tp1, -1)
        policy_tp1 = torch.exp(log_pis_tp1)
        # Q-values.
        q_t, _ = model.get_q_values(model_out_t)
        # Target Q-values.
        q_tp1, _ = target_model.get_q_values(target_model_out_tp1)
        if policy.config["twin_q"]:
            twin_q_t, _ = model.get_twin_q_values(model_out_t)
            twin_q_tp1, _ = target_model.get_twin_q_values(target_model_out_tp1)
            q_tp1 = torch.min(q_tp1, twin_q_tp1)
        q_tp1 -= alpha * log_pis_tp1

        # Actually selected Q-values (from the actions batch).
        one_hot = F.one_hot(
            train_batch[SampleBatch.ACTIONS].long(), num_classes=q_t.size()[-1]
        )
        q_t_selected = torch.sum(q_t * one_hot, dim=-1)
        if policy.config["twin_q"]:
            twin_q_t_selected = torch.sum(twin_q_t * one_hot, dim=-1)
        # Discrete case: "Best" means weighted by the policy (prob) outputs.
        q_tp1_best = torch.sum(torch.mul(policy_tp1, q_tp1), dim=-1)
        q_tp1_best_masked = (
            1.0 - train_batch[SampleBatch.TERMINATEDS].float()
        ) * q_tp1_best
    # Continuous actions case.
    else:
        # Sample single actions from distribution.
        action_dist_class = _get_dist_class(policy, policy.config, policy.action_space)
        action_dist_inputs_t, _ = model.get_action_model_outputs(model_out_t)
        action_dist_t = action_dist_class(action_dist_inputs_t, model)
        policy_t = (
            action_dist_t.sample()
            if not deterministic
            else action_dist_t.deterministic_sample()
        )
        log_pis_t = torch.unsqueeze(action_dist_t.logp(policy_t), -1)
        action_dist_inputs_tp1, _ = model.get_action_model_outputs(model_out_tp1)
        action_dist_tp1 = action_dist_class(action_dist_inputs_tp1, model)
        policy_tp1 = (
            action_dist_tp1.sample()
            if not deterministic
            else action_dist_tp1.deterministic_sample()
        )
        log_pis_tp1 = torch.unsqueeze(action_dist_tp1.logp(policy_tp1), -1)

        # Q-values for the actually selected actions.
        q_t, _ = model.get_q_values(model_out_t, train_batch[SampleBatch.ACTIONS])
        if policy.config["twin_q"]:
            twin_q_t, _ = model.get_twin_q_values(
                model_out_t, train_batch[SampleBatch.ACTIONS]
            )

        # Q-values for current policy in given current state.
        q_t_det_policy, _ = model.get_q_values(model_out_t, policy_t)
        if policy.config["twin_q"]:
            twin_q_t_det_policy, _ = model.get_twin_q_values(model_out_t, policy_t)
            q_t_det_policy = torch.min(q_t_det_policy, twin_q_t_det_policy)

        # Target q network evaluation.
        q_tp1, _ = target_model.get_q_values(target_model_out_tp1, policy_tp1)
        if policy.config["twin_q"]:
            twin_q_tp1, _ = target_model.get_twin_q_values(
                target_model_out_tp1, policy_tp1
            )
            # Take min over both twin-NNs.
            q_tp1 = torch.min(q_tp1, twin_q_tp1)

        q_t_selected = torch.squeeze(q_t, dim=-1)
        if policy.config["twin_q"]:
            twin_q_t_selected = torch.squeeze(twin_q_t, dim=-1)
        q_tp1 -= alpha * log_pis_tp1

        q_tp1_best = torch.squeeze(input=q_tp1, dim=-1)
        q_tp1_best_masked = (
            1.0 - train_batch[SampleBatch.TERMINATEDS].float()
        ) * q_tp1_best

    # compute RHS of bellman equation
    q_t_selected_target = (
        train_batch[SampleBatch.REWARDS]
        + (policy.config["gamma"] ** policy.config["n_step"]) * q_tp1_best_masked
    ).detach()

    # Compute the TD-error (potentially clipped).
    base_td_error = torch.abs(q_t_selected - q_t_selected_target)
    if policy.config["twin_q"]:
        twin_td_error = torch.abs(twin_q_t_selected - q_t_selected_target)
        td_error = 0.5 * (base_td_error + twin_td_error)
    else:
        td_error = base_td_error

    critic_loss = [torch.mean(train_batch[PRIO_WEIGHTS] * huber_loss(base_td_error))]
    if policy.config["twin_q"]:
        critic_loss.append(
            torch.mean(train_batch[PRIO_WEIGHTS] * huber_loss(twin_td_error))
        )

    # Alpha- and actor losses.
    # Note: In the papers, alpha is used directly, here we take the log.
    # Discrete case: Multiply the action probs as weights with the original
    # loss terms (no expectations needed).
    if model.discrete:
        weighted_log_alpha_loss = policy_t.detach() * (
            -model.log_alpha * (log_pis_t + model.target_entropy).detach()
        )
        # Sum up weighted terms and mean over all batch items.
        alpha_loss = torch.mean(torch.sum(weighted_log_alpha_loss, dim=-1))
        # Actor loss.
        actor_loss = torch.mean(
            torch.sum(
                torch.mul(
                    # NOTE: No stop_grad around policy output here
                    # (compare with q_t_det_policy for continuous case).
                    policy_t,
                    alpha.detach() * log_pis_t - q_t.detach(),
                ),
                dim=-1,
            )
        )
    else:
        alpha_loss = -torch.mean(
            model.log_alpha * (log_pis_t + model.target_entropy).detach()
        )
        # Note: Do not detach q_t_det_policy here b/c is depends partly
        # on the policy vars (policy sample pushed through Q-net).
        # However, we must make sure `actor_loss` is not used to update
        # the Q-net(s)' variables.
        actor_loss = torch.mean(alpha.detach() * log_pis_t - q_t_det_policy)

        #set to None if no divergent_actions parameter
        divergent_actions = getattr(policy, 'divergent_actions',None)
        kl_loss_coef = getattr(policy, 'kl_loss_coef',None)
        if divergent_actions is not None:
            for divergent_policy,divergent_action_dist in divergent_actions.items():
                kl = action_dist_t.kl(divergent_action_dist)
                kl_loss = (15-torch.mean(kl))**2 * kl_loss_coef
                #print('actor loss', actor_loss)
                #print('kl loss', kl_loss_coef * kl_loss)
                actor_loss += kl_loss #* kl_loss
            #delete divergent action policies
            delattr(policy, 'divergent_actions')

    # Store values for stats function in model (tower), such that for
    # multi-GPU, we do not override them during the parallel loss phase.
    model.tower_stats["q_t"] = q_t
    model.tower_stats["policy_t"] = policy_t
    model.tower_stats["log_pis_t"] = log_pis_t
    model.tower_stats["actor_loss"] = actor_loss
    model.tower_stats["critic_loss"] = critic_loss
    model.tower_stats["alpha_loss"] = alpha_loss

    # TD-error tensor in final stats
    # will be concatenated and retrieved for each individual batch item.
    model.tower_stats["td_error"] = td_error

    # Return all loss terms corresponding to our optimizers.
    return tuple([actor_loss] + critic_loss + [alpha_loss])

# Build a child class of `TorchPolicy`, given the custom functions defined
# above.
CustomSACTorchPolicy = build_policy_class(
    name="CustomSACTorchPolicy",
    framework="torch",
    loss_fn=custom_actor_critic_loss,
    get_default_config=lambda: ray.rllib.algorithms.sac.sac.SACConfig(),
    stats_fn=stats,
    postprocess_fn=postprocess_trajectory,
    extra_grad_process_fn=apply_grad_clipping,
    optimizer_fn=optimizer_fn,
    validate_spaces=validate_spaces,
    before_loss_init=setup_late_mixins,
    make_model_and_action_dist=build_sac_model_and_action_dist,
    extra_learn_fetches_fn=concat_multi_gpu_td_errors,
    mixins=[TargetNetworkMixin, ComputeTDErrorMixin],
    action_distribution_fn=action_distribution_fn,
)
