from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterable

import torch

from instinct_rl.modules.actor_critic import ActorCritic
from instinct_rl.modules.mlp import MlpModel
from instinct_rl.utils.utils import get_subobs_by_components, get_subobs_size


class ContrastiveActorCritic(ActorCritic):
    """Actor-critic with deployable and privileged representation encoders.

    The actor consumes only a latent computed from deployable policy observations.
    The critic consumes a latent computed from selected privileged observations plus
    the remaining critic context. The privileged latent is therefore trained by the
    value loss and can serve as the target for actor representation alignment.
    """

    def __init__(
        self,
        obs_format: Dict[str, Dict[str, tuple]],
        num_actions: int,
        actor_obs_components: Iterable[str] | None = None,
        critic_target_components: Iterable[str] = (),
        critic_context_components: Iterable[str] | None = None,
        response_target_components: Iterable[str] | None = None,
        actor_encoder_hidden_dims: Iterable[int] = (256, 256),
        critic_encoder_hidden_dims: Iterable[int] = (256, 256),
        response_encoder_hidden_dims: Iterable[int] = (256, 256),
        representation_dim: int = 128,
        projection_hidden_dim: int = 128,
        projection_dim: int = 64,
        **kwargs,
    ):
        policy_segments = OrderedDict(obs_format["policy"])
        critic_segments = OrderedDict(obs_format.get("critic", obs_format["policy"]))
        actor_components = tuple(actor_obs_components or policy_segments.keys())
        critic_target_components = tuple(critic_target_components)
        if not critic_target_components:
            raise ValueError("critic_target_components must contain at least one observation component.")
        critic_context_components = tuple(
            critic_context_components
            if critic_context_components is not None
            else (name for name in critic_segments if name not in critic_target_components)
        )
        response_components = tuple(response_target_components or actor_components)

        self._validate_components("actor_obs_components", actor_components, policy_segments)
        self._validate_components("critic_target_components", critic_target_components, critic_segments)
        self._validate_components("critic_context_components", critic_context_components, critic_segments)
        self._validate_components("response_target_components", response_components, policy_segments)

        actor_input_size = get_subobs_size(policy_segments, actor_components)
        critic_target_size = get_subobs_size(critic_segments, critic_target_components)
        critic_context_size = get_subobs_size(critic_segments, critic_context_components)
        response_target_size = get_subobs_size(policy_segments, response_components)

        self._policy_obs_segments = policy_segments
        self._critic_obs_segments = critic_segments
        self.actor_obs_components = actor_components
        self.critic_target_components = critic_target_components
        self.critic_context_components = critic_context_components
        self.response_target_components = response_components
        self.representation_dim = representation_dim

        embedded_obs_format = {
            "policy": OrderedDict(actor_representation=(representation_dim,)),
            "critic": OrderedDict(
                critic_representation=(representation_dim,),
                critic_context=(critic_context_size,),
            ),
        }
        super().__init__(
            obs_format=embedded_obs_format,
            num_actions=num_actions,
            **kwargs,
        )

        self.actor_encoder = MlpModel(
            input_size=actor_input_size,
            hidden_sizes=list(actor_encoder_hidden_dims),
            output_size=representation_dim,
        )
        self.critic_encoder = MlpModel(
            input_size=critic_target_size,
            hidden_sizes=list(critic_encoder_hidden_dims),
            output_size=representation_dim,
        )
        self.response_encoder = MlpModel(
            input_size=response_target_size,
            hidden_sizes=list(response_encoder_hidden_dims),
            output_size=representation_dim,
        )
        self.actor_projector = MlpModel(
            input_size=representation_dim,
            hidden_sizes=[projection_hidden_dim],
            output_size=projection_dim,
        )
        self.target_projector = MlpModel(
            input_size=representation_dim,
            hidden_sizes=[projection_hidden_dim],
            output_size=projection_dim,
        )

    @staticmethod
    def _validate_components(label: str, components: tuple[str, ...], segments: OrderedDict) -> None:
        missing = [name for name in components if name not in segments]
        if missing:
            raise ValueError(f"{label} contains unknown components {missing}; available={list(segments)}")

    def _policy_subobs(self, observations: torch.Tensor, components: tuple[str, ...]) -> torch.Tensor:
        return get_subobs_by_components(
            observations,
            component_names=components,
            obs_segments=self._policy_obs_segments,
        )

    def _critic_subobs(self, observations: torch.Tensor, components: tuple[str, ...]) -> torch.Tensor:
        if not components:
            return observations.new_empty((*observations.shape[:-1], 0))
        return get_subobs_by_components(
            observations,
            component_names=components,
            obs_segments=self._critic_obs_segments,
        )

    def encode_actor(self, observations: torch.Tensor) -> torch.Tensor:
        actor_obs = self._policy_subobs(observations, self.actor_obs_components)
        return self.actor_encoder(actor_obs)

    def encode_privileged_target(self, critic_observations: torch.Tensor) -> torch.Tensor:
        target_obs = self._critic_subobs(critic_observations, self.critic_target_components)
        return self.critic_encoder(target_obs)

    def encode_response_target(self, observations: torch.Tensor) -> torch.Tensor:
        response_obs = self._policy_subobs(observations, self.response_target_components)
        return self.response_encoder(response_obs)

    def project_actor_representation(self, representation: torch.Tensor) -> torch.Tensor:
        return self.actor_projector(representation)

    def project_target_representation(self, representation: torch.Tensor) -> torch.Tensor:
        return self.target_projector(representation)

    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        return super().act(self.encode_actor(observations), **kwargs)

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        return super().act_inference(self.encode_actor(observations))

    def evaluate(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
        target_latent = self.encode_privileged_target(critic_observations)
        context = self._critic_subobs(critic_observations, self.critic_context_components)
        return super().evaluate(torch.cat((target_latent, context), dim=-1), **kwargs)

    @property
    def obs_segments(self):
        return self._policy_obs_segments

    @property
    def critic_obs_segments(self):
        return self._critic_obs_segments
