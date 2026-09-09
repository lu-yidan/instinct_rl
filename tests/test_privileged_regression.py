from collections import OrderedDict

import torch

from instinct_rl.algorithms import PrivilegedRegressionPPO, PrivilegedRegressionWasabiPPO
from instinct_rl.modules import PrivilegedRegressionActorCritic


def _obs_format():
    return {
        "policy": OrderedDict(proprio=(12,), command=(3,)),
        "critic": OrderedDict(proprio=(12,), command=(3,), terrain=(20,)),
        "amp_policy": OrderedDict(state=(10,)),
        "amp_reference": OrderedDict(state=(10,)),
    }


def _model():
    return PrivilegedRegressionActorCritic(
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
        regression_hidden_dims=(12,),
    )


def _fill_rollout(algorithm, steps=3, with_amp=False):
    policy_obs = torch.randn(8, 15)
    critic_obs = torch.randn(8, 35)
    for step in range(steps):
        algorithm.act(policy_obs, critic_obs)
        next_policy_obs = torch.randn(8, 15)
        next_critic_obs = torch.randn(8, 35)
        dones = torch.zeros(8, dtype=torch.bool)
        if step == 1:
            dones[0] = True
        observations = {}
        if with_amp:
            observations = {
                "amp_policy": torch.randn(8, 10),
                "amp_reference": torch.randn(8, 10),
            }
        algorithm.process_env_step(
            rewards=torch.randn(8, 1),
            dones=dones,
            infos={"observations": observations, "step": {}},
            next_obs=next_policy_obs,
            next_critic_obs=next_critic_obs,
        )
        policy_obs = next_policy_obs
        critic_obs = next_critic_obs
    algorithm.compute_returns(critic_obs)


def test_privileged_regression_model_predicts_raw_target_shape():
    model = _model()
    policy_obs = torch.randn(5, 15)
    critic_obs = torch.randn(5, 35)

    actor_latent = model.encode_actor(policy_obs)
    assert model.predict_privileged_target(actor_latent).shape == (5, 20)
    assert model.privileged_regression_target(critic_obs).shape == (5, 20)
    assert model.act(policy_obs).shape == (5, 4)
    assert sum(parameter.numel() for parameter in model.actor_projector.parameters()) == 0
    assert sum(parameter.numel() for parameter in model.target_projector.parameters()) == 0


def test_current_and_next_privileged_regression_updates_are_finite():
    for target_timestep in ("current", "next"):
        model = _model()
        algorithm = PrivilegedRegressionPPO(
            model,
            num_learning_epochs=1,
            num_mini_batches=1,
            privileged_regression_target_timestep=target_timestep,
        )
        algorithm.init_storage(8, 3, _obs_format(), 4)
        _fill_rollout(algorithm)
        losses, stats = algorithm.update(0)

        assert torch.isfinite(losses["privileged_regression_loss"])
        assert torch.isfinite(losses["total_loss"])
        assert torch.isfinite(stats["privileged_regression_rmse"])
        expected_samples = 23 if target_timestep == "next" else 24
        assert stats["privileged_regression_samples"] == expected_samples


def test_privileged_regression_wasabi_composes_both_updates():
    model = _model()
    algorithm = PrivilegedRegressionWasabiPPO(
        model,
        num_learning_epochs=1,
        num_mini_batches=1,
        discriminator_kwargs={"hidden_sizes": [16]},
        discriminator_gradient_penalty_coef=0.0,
    )
    algorithm.init_storage(8, 2, _obs_format(), 4)
    _fill_rollout(algorithm, steps=2, with_amp=True)
    losses, _ = algorithm.update(0)

    assert torch.isfinite(losses["privileged_regression_loss"])
    assert torch.isfinite(losses["discriminator_loss"])
