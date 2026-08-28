from collections import OrderedDict

import torch

from instinct_rl.algorithms import ContrastivePPO, ContrastiveWasabiPPO
from instinct_rl.modules import ContrastiveActorCritic
from instinct_rl.storage import ContrastiveRolloutStorage


def _obs_format():
    return {
        "policy": OrderedDict(proprio=(12,), command=(3,)),
        "critic": OrderedDict(proprio=(12,), command=(3,), terrain=(20,)),
        "amp_policy": OrderedDict(state=(10,)),
        "amp_reference": OrderedDict(state=(10,)),
    }


def _model():
    return ContrastiveActorCritic(
        obs_format=_obs_format(),
        num_actions=4,
        critic_target_components=("terrain",),
        actor_hidden_dims=(32, 16),
        critic_hidden_dims=(32, 16),
        actor_encoder_hidden_dims=(32,),
        critic_encoder_hidden_dims=(32,),
        response_encoder_hidden_dims=(32,),
        representation_dim=16,
        projection_hidden_dim=16,
        projection_dim=8,
    )


def test_contrastive_actor_critic_uses_raw_observation_interfaces():
    model = _model()
    policy_obs = torch.randn(5, 15)
    critic_obs = torch.randn(5, 35)

    assert model.act(policy_obs).shape == (5, 4)
    assert model.act_inference(policy_obs).shape == (5, 4)
    assert model.evaluate(critic_obs).shape == (5, 1)
    assert model.encode_actor(policy_obs).shape == (5, 16)
    assert model.encode_privileged_target(critic_obs).shape == (5, 16)
    assert model.encode_response_target(policy_obs).shape == (5, 16)


def test_current_and_next_contrastive_updates_are_finite():
    for target_source, target_timestep, stop_target_gradient in (
        ("critic", "current", True),
        ("critic", "next", True),
        ("policy", "next", True),
    ):
        model = _model()
        algorithm = ContrastivePPO(
            model,
            num_learning_epochs=1,
            num_mini_batches=1,
            contrastive_loss_coef=0.1,
            contrastive_target_source=target_source,
            contrastive_target_timestep=target_timestep,
            contrastive_stop_target_gradient=stop_target_gradient,
        )
        algorithm.init_storage(8, 3, _obs_format(), 4)

        policy_obs = torch.randn(8, 15)
        critic_obs = torch.randn(8, 35)
        for step in range(3):
            algorithm.act(policy_obs, critic_obs)
            next_policy_obs = torch.randn(8, 15)
            next_critic_obs = torch.randn(8, 35)
            dones = torch.zeros(8, dtype=torch.bool)
            if step == 1:
                dones[0] = True
            algorithm.process_env_step(
                rewards=torch.randn(8, 1),
                dones=dones,
                infos={"observations": {}, "step": {}},
                next_obs=next_policy_obs,
                next_critic_obs=next_critic_obs,
            )
            policy_obs = next_policy_obs
            critic_obs = next_critic_obs

        algorithm.compute_returns(critic_obs)
        losses, stats = algorithm.update(0)

        assert torch.isfinite(losses["contrastive_loss"])
        assert torch.isfinite(losses["total_loss"])
        assert 0.0 <= stats["contrastive_top1"] <= 1.0


def test_no_aux_uses_the_same_algorithm_without_contrastive_loss():
    model = _model()
    algorithm = ContrastivePPO(model, contrastive_loss_coef=0.0)

    assert isinstance(algorithm, ContrastivePPO)
    assert algorithm.contrastive_loss_coef == 0.0


def test_terminal_next_observation_is_not_a_valid_contrastive_target():
    storage = ContrastiveRolloutStorage(
        num_envs=2,
        num_transitions_per_env=1,
        obs_shape=(15,),
        critic_obs_shape=(35,),
        actions_shape=(4,),
    )
    storage.dones[0, 0] = True

    minibatch = storage.get_minibatch_from_selection(
        torch.tensor([0, 0]),
        torch.tensor([0, 1]),
    )

    assert not minibatch.next_valid[0]
    assert minibatch.next_valid[1]


def test_contrastive_wasabi_composes_amp_and_contrastive_updates():
    model = _model()
    algorithm = ContrastiveWasabiPPO(
        model,
        num_learning_epochs=1,
        num_mini_batches=1,
        contrastive_loss_coef=0.1,
        discriminator_kwargs={"hidden_sizes": [16]},
        discriminator_gradient_penalty_coef=0.0,
    )
    algorithm.init_storage(8, 2, _obs_format(), 4)

    policy_obs = torch.randn(8, 15)
    critic_obs = torch.randn(8, 35)
    for _ in range(2):
        algorithm.act(policy_obs, critic_obs)
        next_policy_obs = torch.randn(8, 15)
        next_critic_obs = torch.randn(8, 35)
        algorithm.process_env_step(
            rewards=torch.randn(8, 1),
            dones=torch.zeros(8, dtype=torch.bool),
            infos={
                "observations": {
                    "amp_policy": torch.randn(8, 10),
                    "amp_reference": torch.randn(8, 10),
                },
                "step": {},
            },
            next_obs=next_policy_obs,
            next_critic_obs=next_critic_obs,
        )
        policy_obs = next_policy_obs
        critic_obs = next_critic_obs

    algorithm.compute_returns(critic_obs)
    losses, _ = algorithm.update(0)

    assert torch.isfinite(losses["contrastive_loss"])
    assert torch.isfinite(losses["discriminator_loss"])
