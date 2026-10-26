# GRPO checkpoint resume

The synchronous GRPO recipe saves checkpoints at optimizer-step boundaries.
The driver and training workers must share the checkpoint filesystem.

## Usage

Start training:

```bash
cd cookbook/rl/grpo
sh grpo.sh --output-dir /shared/grpo-run --save-steps 10
```

Resume in a fresh process:

```bash
sh grpo.sh \
    --output-dir /shared/grpo-resumed \
    --save-steps 10 \
    --resume-from-checkpoint /shared/grpo-run/grpo-gsm8k-checkpoint-10
```

Repeat the original model, batch size, generation, seed, learning rate,
topology and maximum-step overrides. The recorded configuration must match,
including max_steps. Full continuation requires resume_only_model and
ignore_data_skip to remain disabled.

## Restored state

- Model or adapter weights, plus training state supported by the backend.
- Optimizer-step count and dataloader progress.
- Remaining rollout inputs, old log probabilities, advantages and metrics.
- Driver Python RNG and per-rank Transformers training RNG.

Each rollout receives an explicit seed drawn from the saved driver RNG.
A fresh sampler receives base weights and the current LoRA adapter before
sampling.

## Limitations

Recovery starts from the latest completed checkpoint. Work performed after
that checkpoint is repeated. In-flight generation and partially accumulated
gradients are not saved.

The RL sidecar, rl_state.pkl, is atomically replaced after synchronous model
saving succeeds. The complete checkpoint directory is not transactional,
and the helper does not independently verify every model file.
Full RL recovery requires this sidecar. Use a new output directory when
resuming to avoid checkpoint-name collisions.

Keep datasets, reward implementations and dependencies unchanged; the
configuration check does not fingerprint all of them. Exact numerical
reproducibility also depends on software, hardware and execution settings.

Load checkpoints only from trusted runs: the sidecar uses pickle.
Stored rollouts increase checkpoint size and may contain training data.

Legacy Transformers checkpoints can use rng_state.pt. New per-rank
checkpoints require the current rank's RNG file. Legacy model checkpoints
without an RL sidecar cannot provide full GRPO recovery.

Asynchronous RL, other recipes, Megatron continuation and full multi-rank
RL continuation require separate validation.

## Validation

A short Qwen2.5-0.5B-Instruct LoRA experiment used one NPU training worker
and one NPU vLLM sampler. Synthetic nonzero rewards ensured actual updates.

An uninterrupted eight-step run was compared with a fresh process resumed
from step one, covering two rollout batches. At steps four and eight,
adapter weights, optimizer, scheduler, training RNG and RL controller state
matched exactly. Remaining inputs, old log probabilities and advantages
in the second batch also matched. Between steps one and eight, 336 adapter
tensors changed.

This validates continuation in the tested configuration, rather than
GSM8K reward improvement.
