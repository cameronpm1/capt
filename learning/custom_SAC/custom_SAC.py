import math
import time
import torch
import logging
import numpy as np
from typing import Any, Dict, Optional, Tuple, Type, Union

from ray.rllib.policy.policy import Policy
from ray.rllib.algorithms.sac.sac import SAC
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.utils.annotations import override
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import  ResultDict, PolicyID
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.algorithms.dqn.dqn import calculate_rr_weights
from ray.rllib.algorithms.sac.sac_tf_policy import SACTFPolicy
from ray.rllib.utils.typing import RLModuleSpec, SampleBatchType
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.utils.replay_buffers.utils import sample_min_n_steps_from_buffer
from ray.rllib.execution.train_ops import multi_gpu_train_one_step, train_one_step
from ray.rllib.utils.replay_buffers.utils import update_priorities_in_replay_buffer
from ray.rllib.utils.metrics import (
    ALL_MODULES,
    ENV_RUNNER_RESULTS,
    ENV_RUNNER_SAMPLING_TIMER,
    LAST_TARGET_UPDATE_TS,
    LEARNER_RESULTS,
    LEARNER_UPDATE_TIMER,
    LOAD_BATCH_TIMER,
    LEARN_ON_BATCH_TIMER,
    NUM_AGENT_STEPS_SAMPLED,
    NUM_AGENT_STEPS_TRAINED,
    NUM_AGENT_STEPS_SAMPLED_LIFETIME,
    NUM_ENV_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
    NUM_ENV_STEPS_TRAINED,
    NUM_ENV_STEPS_TRAINED_LIFETIME,
    NUM_EPISODES,
    NUM_EPISODES_LIFETIME,
    NUM_MODULE_STEPS_SAMPLED,
    NUM_MODULE_STEPS_SAMPLED_LIFETIME,
    NUM_MODULE_STEPS_TRAINED,
    NUM_MODULE_STEPS_TRAINED_LIFETIME,
    NUM_TARGET_UPDATES,
    REPLAY_BUFFER_SAMPLE_TIMER,
    REPLAY_BUFFER_UPDATE_PRIOS_TIMER,
    SAMPLE_TIMER,
    SYNCH_WORKER_WEIGHTS_TIMER,
    TD_ERROR_KEY,
    TIMERS,
)

from learning.custom_SAC.custom_sac_torch_policy import _get_dist_class

logger = logging.getLogger(__name__)

class custom_SACConfig(SACConfig):
    """
    custom config module for custom_SAC
    """

    def __init__(self):
        super().__init__(algo_class=custom_SAC)

class custom_SAC(SAC):
    
    """
    Custom SAC class for overwritting train step

    handles multi-agent training in parallel
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    @override(SAC)
    def get_default_policy_class(
        cls, config: AlgorithmConfig
    ) -> Optional[Type[Policy]]:
        if config["framework"] == "torch":
            from learning.custom_SAC.custom_sac_torch_policy import CustomSACTorchPolicy

            return CustomSACTorchPolicy
        else:
            return SACTFPolicy
        
    def get_policy(self, policy_id: PolicyID = DEFAULT_POLICY_ID) -> Policy:
        """Return policy for the specified id, or None.

        Args:
            policy_id: ID of the which policy model to return.
        """
        return self.workers.local_worker().get_policy(policy_id)
    
    def kl_loss_schedule(
        self,
        ts: int
    ):
        start_val = 0.1
        stop_val = 0
        ts_begin = 0
        ts_end = 5e6
        if ts > ts_begin and ts < ts_end:
            return (1-(ts-ts_begin)/(ts_end-ts_begin)) * (start_val-stop_val) + stop_val
        elif ts > ts_end:
            return stop_val
        else:
            return start_val

    @override(SAC)
    def _training_step_old_and_hybrid_api_stack(self) -> ResultDict:
        """Training step for the old and hybrid training stacks.

        More specifically this training step relies on `RolloutWorker`.
        """
        train_results = {}

        # We alternate between storing new samples and sampling and training
        store_weight, sample_and_train_weight = calculate_rr_weights(self.config)

        for _ in range(store_weight):
            # Sample (MultiAgentBatch) from workers.
            with self._timers[SAMPLE_TIMER]:
                new_sample_batch: SampleBatchType = synchronous_parallel_sample(
                    worker_set=self.workers,
                    concat=True,
                    sample_timeout_s=self.config.sample_timeout_s,
                )

            # Return early if all our workers failed.
            if not new_sample_batch:
                return {}

            # Update counters
            self._counters[NUM_AGENT_STEPS_SAMPLED] += new_sample_batch.agent_steps()
            self._counters[NUM_ENV_STEPS_SAMPLED] += new_sample_batch.env_steps()

            # Store new samples in replay buffer.
            self.local_replay_buffer.add(new_sample_batch)

        global_vars = {
            "timestep": self._counters[NUM_ENV_STEPS_SAMPLED],
        }

        # Update target network every `target_network_update_freq` sample steps.
        cur_ts = self._counters[
            (
                NUM_AGENT_STEPS_SAMPLED
                if self.config.count_steps_by == "agent_steps"
                else NUM_ENV_STEPS_SAMPLED
            )
        ]

        if cur_ts > self.config.num_steps_sampled_before_learning_starts:
            for _ in range(sample_and_train_weight):
                # Sample training batch (MultiAgentBatch) from replay buffer.
                train_batch = sample_min_n_steps_from_buffer(
                    self.local_replay_buffer,
                    self.config.train_batch_size,
                    count_by_agent_steps=self.config.count_steps_by == "agent_steps",
                )

                # Postprocess batch before we learn on it
                post_fn = self.config.get("before_learn_on_batch") or (lambda b, *a: b)
                train_batch = post_fn(train_batch, self.workers, self.config)
                
                #set kl_loss_val
                kl_loss_coef = self.kl_loss_schedule(cur_ts)
                #collect parallel policy actions
                for policy_id1 in train_batch.policy_batches.keys():
                    divergent_actions = {}
                    for policy_id2 in train_batch.policy_batches.keys():
                        if policy_id1 == policy_id2:
                            continue
                        else:
                            policy = self.get_policy(policy_id2)
                            model = policy.model
                            with torch.no_grad():
                                #for each policy state, collect parallel policy action dists.
                                model_out, _ = model(
                                    SampleBatch(obs=torch.from_numpy(train_batch[policy_id1][SampleBatch.CUR_OBS]), _is_training=True), [], None
                                )
                                divergent_actions_input, _ = model.get_action_model_outputs(model_out.cuda())
                                action_dist_class = _get_dist_class(policy, policy.config, policy.action_space)
                                divergent_actions[policy_id2] = action_dist_class(divergent_actions_input, model)
                    #set divergent actions policy attribute directly to policy
                    policy = self.get_policy(policy_id1)
                    setattr(policy, 'divergent_actions', divergent_actions)
                    setattr(policy, 'kl_loss_coef', kl_loss_coef)

                # Learn on training batch.
                # Use simple optimizer (only for multi-agent or tf-eager; all other
                # cases should use the multi-GPU optimizer, even if only using 1 GPU)
                if self.config.get("simple_optimizer") is True:
                    train_results = train_one_step(self, train_batch)
                else:
                    train_results = multi_gpu_train_one_step(self, train_batch)

                # Update replay buffer priorities.
                update_priorities_in_replay_buffer(
                    self.local_replay_buffer,
                    self.config,
                    train_batch,
                    train_results,
                )

                last_update = self._counters[LAST_TARGET_UPDATE_TS]
                if cur_ts - last_update >= self.config.target_network_update_freq:
                    to_update = self.workers.local_worker().get_policies_to_train()
                    self.workers.local_worker().foreach_policy_to_train(
                        lambda p, pid, to_update=to_update: (
                            pid in to_update and p.update_target()
                        )
                    )
                    self._counters[NUM_TARGET_UPDATES] += 1
                    self._counters[LAST_TARGET_UPDATE_TS] = cur_ts

                # Update weights and global_vars - after learning on the local worker -
                # on all remote workers.
                with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
                    self.workers.sync_weights(global_vars=global_vars)

        # Return all collected metrics for the iteration.
        return train_results