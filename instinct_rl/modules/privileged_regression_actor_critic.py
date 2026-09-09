from __future__ import annotations

from typing import Dict, Iterable

import torch

from instinct_rl.modules.contrastive_actor_critic import ContrastiveActorCritic
from instinct_rl.modules.mlp import MlpModel
from instinct_rl.utils.utils import get_subobs_size


class PrivilegedRegressionActorCritic(ContrastiveActorCritic):
    """Contrastive-policy architecture with a raw privileged reconstruction head."""

    def __init__(
        self,
        obs_format: Dict[str, Dict[str, tuple]],
        num_actions: int,
        critic_target_components: Iterable[str] = (),
        regression_hidden_dims: Iterable[int] = (128,),
        **kwargs,
    ):
        target_components = tuple(critic_target_components)
        super().__init__(
            obs_format=obs_format,
            num_actions=num_actions,
            critic_target_components=target_components,
            **kwargs,
        )
        target_size = get_subobs_size(obs_format.get("critic", obs_format["policy"]), target_components)
        self.privileged_regressor = MlpModel(
            input_size=self.representation_dim,
            hidden_sizes=list(regression_hidden_dims),
            output_size=target_size,
        )

    def privileged_regression_target(self, critic_observations: torch.Tensor) -> torch.Tensor:
        return self._critic_subobs(critic_observations, self.critic_target_components)

    def predict_privileged_target(self, actor_representation: torch.Tensor) -> torch.Tensor:
        return self.privileged_regressor(actor_representation)
