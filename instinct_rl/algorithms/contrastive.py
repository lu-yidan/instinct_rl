from __future__ import annotations

import torch
import torch.nn.functional as F

from instinct_rl.algorithms.ppo import PPO
from instinct_rl.algorithms.wasabi import WasabiAlgoMixin
from instinct_rl.storage import ContrastiveRolloutStorage
from instinct_rl.utils.utils import get_subobs_size


class ContrastivePPO(PPO):
    """PPO with a configurable contrastive representation objective."""

    def __init__(
        self,
        *args,
        contrastive_loss_coef: float = 0.0,
        contrastive_temperature: float = 0.1,
        contrastive_max_samples: int = 1024,
        contrastive_target_source: str = "critic",
        contrastive_target_timestep: str = "current",
        contrastive_stop_target_gradient: bool = True,
        contrastive_symmetric_loss: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if contrastive_temperature <= 0.0:
            raise ValueError("contrastive_temperature must be positive.")
        if contrastive_max_samples < 2:
            raise ValueError("contrastive_max_samples must be at least 2.")
        if contrastive_target_source not in {"critic", "policy"}:
            raise ValueError("contrastive_target_source must be 'critic' or 'policy'.")
        if contrastive_target_timestep not in {"current", "next"}:
            raise ValueError("contrastive_target_timestep must be 'current' or 'next'.")

        self.contrastive_loss_coef = contrastive_loss_coef
        self.contrastive_temperature = contrastive_temperature
        self.contrastive_max_samples = contrastive_max_samples
        self.contrastive_target_source = contrastive_target_source
        self.contrastive_target_timestep = contrastive_target_timestep
        self.contrastive_stop_target_gradient = contrastive_stop_target_gradient
        self.contrastive_symmetric_loss = contrastive_symmetric_loss

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
        if self.contrastive_loss_coef == 0.0:
            return losses, inter_vars, stats

        actor_obs = minibatch.obs
        if self.contrastive_target_timestep == "next":
            target_obs = minibatch.next_critic_obs if self.contrastive_target_source == "critic" else minibatch.next_obs
            valid = minibatch.next_valid.reshape(-1)
        else:
            target_obs = minibatch.critic_obs if self.contrastive_target_source == "critic" else minibatch.obs
            valid = torch.ones(actor_obs.shape[:-1], dtype=torch.bool, device=actor_obs.device).reshape(-1)

        actor_obs = actor_obs.reshape(-1, actor_obs.shape[-1])[valid]
        target_obs = target_obs.reshape(-1, target_obs.shape[-1])[valid]
        if actor_obs.shape[0] > self.contrastive_max_samples:
            selected = torch.randperm(actor_obs.shape[0], device=actor_obs.device)[: self.contrastive_max_samples]
            actor_obs = actor_obs[selected]
            target_obs = target_obs[selected]
        if actor_obs.shape[0] < 2:
            return losses, inter_vars, stats

        actor_latent = self.actor_critic.encode_actor(actor_obs)
        if self.contrastive_target_source == "critic":
            target_latent = self.actor_critic.encode_privileged_target(target_obs)
        else:
            target_latent = self.actor_critic.encode_response_target(target_obs)
        if self.contrastive_stop_target_gradient:
            target_latent = target_latent.detach()

        actor_projection = F.normalize(
            self.actor_critic.project_actor_representation(actor_latent),
            dim=-1,
        )
        target_projection = F.normalize(
            self.actor_critic.project_target_representation(target_latent),
            dim=-1,
        )
        logits = actor_projection @ target_projection.transpose(0, 1)
        logits = logits / self.contrastive_temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        contrastive_loss = F.cross_entropy(logits, labels)
        if self.contrastive_symmetric_loss:
            contrastive_loss = 0.5 * (
                contrastive_loss + F.cross_entropy(logits.transpose(0, 1), labels)
            )

        losses["contrastive_loss"] = contrastive_loss
        stats["contrastive_top1"] = (logits.argmax(dim=-1) == labels).float().mean()
        stats["contrastive_samples"] = logits.new_tensor(logits.shape[0])
        return losses, inter_vars, stats


class ContrastiveWasabiPPO(WasabiAlgoMixin, ContrastivePPO):
    """Contrastive PPO with the existing Wasabi/AMP reward and discriminator."""

    pass
