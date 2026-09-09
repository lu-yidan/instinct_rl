from __future__ import annotations

import torch
import torch.nn.functional as F

from instinct_rl.algorithms.ppo import PPO
from instinct_rl.algorithms.wasabi import WasabiAlgoMixin
from instinct_rl.storage import ContrastiveRolloutStorage
from instinct_rl.utils.utils import get_subobs_size


class PrivilegedRegressionPPO(PPO):
    """PPO with a stopped-gradient raw privileged reconstruction objective."""

    def __init__(
        self,
        *args,
        privileged_regression_loss_coef: float = 1.0,
        privileged_regression_max_samples: int = 1024,
        privileged_regression_target_timestep: str = "next",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if privileged_regression_max_samples < 1:
            raise ValueError("privileged_regression_max_samples must be positive.")
        if privileged_regression_target_timestep not in {"current", "next"}:
            raise ValueError("privileged_regression_target_timestep must be 'current' or 'next'.")
        self.privileged_regression_loss_coef = privileged_regression_loss_coef
        self.privileged_regression_max_samples = privileged_regression_max_samples
        self.privileged_regression_target_timestep = privileged_regression_target_timestep

    def init_storage(self, num_envs, num_transitions_per_env, obs_format, num_actions, num_rewards=1):
        self.transition = ContrastiveRolloutStorage.Transition()
        obs_size = get_subobs_size(obs_format["policy"])
        critic_obs_size = get_subobs_size(obs_format.get("critic")) if "critic" in obs_format else None
        self.storage = ContrastiveRolloutStorage(
            num_envs,
            num_transitions_per_env,
            [obs_size],
            [critic_obs_size],
            [num_actions],
            num_rewards=num_rewards,
            device=self.device,
        )

    def process_env_step(self, rewards, dones, infos, next_obs, next_critic_obs):
        self.transition.next_observations = next_obs
        self.transition.next_critic_observations = next_critic_obs
        return super().process_env_step(rewards, dones, infos, next_obs, next_critic_obs)

    def compute_losses(self, minibatch):
        losses, inter_vars, stats = super().compute_losses(minibatch)
        if self.privileged_regression_loss_coef == 0.0:
            return losses, inter_vars, stats

        actor_obs = minibatch.obs
        if self.privileged_regression_target_timestep == "next":
            target_obs = minibatch.next_critic_obs
            valid = minibatch.next_valid.reshape(-1)
        else:
            target_obs = minibatch.critic_obs
            valid = torch.ones(actor_obs.shape[:-1], dtype=torch.bool, device=actor_obs.device).reshape(-1)

        actor_obs = actor_obs.reshape(-1, actor_obs.shape[-1])[valid]
        target_obs = target_obs.reshape(-1, target_obs.shape[-1])[valid]
        if actor_obs.shape[0] > self.privileged_regression_max_samples:
            selected = torch.randperm(actor_obs.shape[0], device=actor_obs.device)[
                : self.privileged_regression_max_samples
            ]
            actor_obs = actor_obs[selected]
            target_obs = target_obs[selected]
        if actor_obs.shape[0] == 0:
            return losses, inter_vars, stats

        actor_latent = self.actor_critic.encode_actor(actor_obs)
        prediction = self.actor_critic.predict_privileged_target(actor_latent)
        target = self.actor_critic.privileged_regression_target(target_obs).detach()
        regression_loss = F.mse_loss(prediction, target)

        losses["privileged_regression_loss"] = regression_loss
        stats["privileged_regression_rmse"] = regression_loss.detach().sqrt()
        stats["privileged_regression_samples"] = regression_loss.new_tensor(actor_obs.shape[0])
        return losses, inter_vars, stats


class PrivilegedRegressionWasabiPPO(WasabiAlgoMixin, PrivilegedRegressionPPO):
    """Privileged-regression PPO with the existing Wasabi/AMP objective."""

    pass
