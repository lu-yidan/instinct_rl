# Contrastive PPO

`ContrastivePPO` provides the matched representation-learning objectives used by
the Adam SP CRA revision study. It extends the existing PPO path instead of
forking the runner.

## Model boundary

`ContrastiveActorCritic` contains three representation encoders:

- `actor_encoder`: consumes only deployable policy observations and supplies the
  actor representation;
- `critic_encoder`: consumes selected privileged critic components, supplies the
  critic representation, and is trained by the value loss;
- `response_encoder`: consumes selected policy observations and supplies a
  successor-proprioception target.

The actor and target representations are passed through separate projection
heads before InfoNCE. When `contrastive_stop_target_gradient` is enabled, the
target encoder is not shaped by InfoNCE. For privileged CRA targets, the critic
encoder remains trained by the value loss.

## Controlled variants

| Variant | Target source | Target time | Stop target gradient |
|---|---|---:|---:|
| NoAux | none | — | — |
| CRA-current | privileged critic observation | `t` | yes |
| CRA-next | privileged critic observation | `t+1` | yes |
| Proprio-next | successor policy response | `t+1` | yes |

NoAux still uses the same policy class, encoders, optimizer, rollout storage,
and actor/critic capacity. Its only difference is
`contrastive_loss_coef=0.0`.

`ContrastiveRolloutStorage` excludes terminal transitions from next-step
InfoNCE because the post-terminal observation belongs to a reset episode.

Proprio-next is a matched target-content control, not a reproduction of
HIMLoco. HIMLoco uses an additional velocity-estimation loss and a
prototype-based swapped-assignment objective; results from this implementation
must not be reported as HIMLoco results.

## Configuration keys

Policy keys:

- `actor_obs_components`
- `critic_target_components`
- `critic_context_components`
- `response_target_components`
- `actor_encoder_hidden_dims`
- `critic_encoder_hidden_dims`
- `response_encoder_hidden_dims`
- `representation_dim`
- `projection_hidden_dim`
- `projection_dim`

Algorithm keys:

- `contrastive_loss_coef`
- `contrastive_temperature`
- `contrastive_max_samples`
- `contrastive_target_source`: `critic` or `policy`
- `contrastive_target_timestep`: `current` or `next`
- `contrastive_stop_target_gradient`
- `contrastive_symmetric_loss`
