# From the local lab to slime and Prime Intellect

The local lab teaches the update and ownership contracts. This track teaches operating the actual frameworks. GPU jobs below have not been run for this repository. Start with the framework's own known example before introducing our task.

## Pin the sources

From the rltune root:

```bash
uv run python scripts/prepare_upstreams.py
```

This fetches detached source checkouts under `.external/`; it neither installs frameworks nor launches jobs. Revisions reviewed on 6 November 2026:

| Stack | Revision | Contract inspected |
|---|---|---|
| [slime](https://github.com/THUDM/slime/tree/79aa41a884ab0bf65222c94418f6357d3c31192b) | `79aa41a` | async custom reward dispatch and Qwen model recipes |
| [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl/tree/c82ef259418aa9d21159380b2b811cbca023befa) | `c82ef25` | reverse-text TOML, inference/rl entrypoints |
| [verifiers](https://github.com/PrimeIntellect-ai/verifiers/tree/897dd5f84bb8745e2f6705bcacf0f919ed8ce40c) | `897dd5f` | v1 TaskData / Task / Taskset, stop and reward decorators |

prime-rl pins its own dependency submodules. Inspect its `deps/verifiers` revision after submodule initialization; do not assume that a separately reviewed latest verifiers revision is the same. The adapter uses the inspected v1 contract. Test its loading in the actual prime-rl environment before launching training. A source pin also does not pin a Docker image: record the image digest when you run it.

## Prime Intellect: build a verifiers environment locally

Current upstream examples use `verifiers.v1`, `TaskData`, `Task`, `Taskset` and reward decorators. This repository follows that interface rather than an older `SingleTurnEnv` recipe.

```bash
# In the rltune checkout. The optional dependency pins the reviewed verifiers source.
uv sync --extra dev --extra prime --locked
uv pip install --no-deps -e integrations/prime
uv run --no-sync pytest -q tests/test_verifiers_integration.py
uv run --no-sync vf-validate rltune-arithmetic -n 2 --only-setup \
  --runtime.type subprocess -o runs/verifiers-validation
```

`--no-sync` preserves the separately installed local taskset package. The `subprocess` runtime keeps this setup check local. The upstream default runtime is Prime-hosted and requires authentication. No model is called by `--only-setup`.

**Locally verified:** 80 train tasks and 20 disjoint evaluation tasks construct using the real installed verifiers library. The integration test invokes the reward with real `vf.Trace` objects, including malformed answers. The validation CLI reports 2/2 valid setup checks and 2/2 valid noop checks on the subprocess runtime. This validates task construction and setup, not inference or gradient updates.

The adapter reads raw reply content rather than `trace.last_reply`, because the latter strips whitespace at this revision. Otherwise an invalid trailing newline would be rewarded differently across the local and framework verifiers. Labels live in task data, outside the prompt. `@vf.stop` limits the task to one turn.

Sources: [taskset guide](https://github.com/PrimeIntellect-ai/verifiers/blob/897dd5f84bb8745e2f6705bcacf0f919ed8ce40c/docs/tasksets.md), [reverse-text implementation](https://github.com/PrimeIntellect-ai/verifiers/blob/897dd5f84bb8745e2f6705bcacf0f919ed8ce40c/environments/reverse_text/reverse_text/taskset.py).

### Operate prime-rl on NVIDIA hardware

Use a separate Linux GPU environment, following the pinned [installation instructions](https://github.com/PrimeIntellect-ai/prime-rl/blob/c82ef259418aa9d21159380b2b811cbca023befa/README.md). The package has CUDA/native dependency requirements; it is not installed into the CPU lab by default.

First establish the upstream reverse-text baseline:

```bash
cd .external/prime-rl
git submodule update --init -- deps/verifiers deps/renderers deps/prime-envs deps/pydantic-config
uv sync --all-extras --all-packages
# Follow the pinned README's CUDA / flash-attention checks before training.
uv run inference --vllm.model Qwen/Qwen3-0.6B
# In a second terminal, same checkout/environment:
uv run rl @ examples/basic/reverse-text/rl.toml
```

The pinned reverse-text training TOML selects `PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT`, while the README's introductory inference command selects base Qwen3-0.6B. Inspect whether your chosen recipe launches its own inference service; align model identity and ports if using a separately started service. Do not train against a mismatched serving policy.

Then install rltune and the taskset into **that** environment without replacing prime-rl's pinned dependencies. From the root rltune checkout:

```bash
uv pip install --python .external/prime-rl/.venv/bin/python --no-deps -e . -e integrations/prime
.external/prime-rl/.venv/bin/vf-validate rltune-arithmetic -n 2 --only-setup \
  --runtime.type subprocess -o runs/prime-taskset-validation
cd .external/prime-rl
uv run --no-sync rl @ ../../integrations/prime/rl.toml
```

Our TOML is adapted from the inspected reverse-text recipe, with a base Qwen3-0.6B model, separate train/eval taskset splits and short completions. Its TOML syntax is checked locally; prime-rl schema validation and GPU execution are pending. Inspect the resolved config, model renderer and GPU allocation using the pinned framework instructions before the last command. Keep logging/authentication settings in your own environment.

**Learning evidence to collect:** rollout group size and effective batch size; inference model identity; behavior and learner version; weight-publication timing; trainer reward/KL; held-out accuracy; checkpoint restore; memory and throughput. Demonstrate that rollout workers serve updated weights after an optimizer update.

### Environments Hub exercise

The local `rltune-arithmetic` package is a starting point for a Hub environment, not a published Hub artifact. Follow the current [Building Environments guide](https://github.com/PrimeIntellect-ai/verifiers/blob/897dd5f84bb8745e2f6705bcacf0f919ed8ce40c/docs/building-environments.md) and [Prime CLI docs](https://docs.primeintellect.ai/). Inspect `prime env --help` in your installed CLI for the current publication workflow. Validate locally, package with a version, and record the environment ID/version and reward contract when you choose to publish. No Hub publishing or hosted job is performed by this repository's setup scripts.

## slime: Megatron learner and SGLang rollout

Use a supported Linux NVIDIA host and the official [environment setup](https://github.com/THUDM/slime/blob/79aa41a884ab0bf65222c94418f6357d3c31192b/docs/en/get_started/quick_start.md). The inspected guide uses CUDA 13 Docker images for Hopper/Blackwell and provides checkpoint conversion/setup steps. Save a concrete image digest, model revision and data hash in your experiment record.

Start with the pinned Qwen3-4B recipe and its documented model/data preparation. Review it before execution: it includes broad process cleanup (`pkill`) and assumes particular filesystem paths and GPU availability. Use a dedicated training container. Do not run an unedited upstream script on a shared host. The rltune preparation script never executes it.

After the unmodified baseline works, export our task:

```bash
# rltune environment, from repository root
uv run python -m integrations.slime.export_data
```

This writes `runs/slime-data/train.jsonl` and `eval.jsonl`, with prompt and label fields. Install rltune into the slime training environment and make the repository importable on every reward worker. For example, mount it as `/workspace/rltune`, install it there with the image's Python, and include `/workspace/rltune` alongside Megatron in the worker `PYTHONPATH`.

Make a copy of the pinned [Qwen3 recipe](https://github.com/THUDM/slime/blob/79aa41a884ab0bf65222c94418f6357d3c31192b/scripts/run-qwen3-4B.sh) inside the slime scripts directory, so its relative model-config sourcing still works. In that copy, replace the data and reward arguments:

```bash
--prompt-data /workspace/rltune/runs/slime-data/train.jsonl
--input-key prompt
--label-key label
--apply-chat-template
--custom-rm-path integrations.slime.reward.reward
--rollout-max-response-len 32
```

Remove the old `--rm-type deepscaler` selection and configure evaluation against the separate eval file. Retain the model's architecture arguments and correctly converted HF/Megatron checkpoints from the working baseline. Start with a short smoke run and an early checkpoint, not the original recipe's 3,000 rollouts and 131,072-token response cap. Keep global batch size divisible by the effective training setup and verify GPU counts/parallelism for your machine.

The hook implements the inspected `async reward(args, sample, **kwargs)` contract. It accepts both a single `Sample` and a list for batched custom-RM dispatch. Labels are expected to be integer strings. Invalid dataset labels raise rather than silently awarding zero. The core verifier is tested locally; actual slime `Sample` dispatch through GPU rollout workers remains pending.

Trace the actual call path in the pinned source:

1. `train.py`: outer training loop and rollout/update sequencing.
2. `slime/rollout/sglang_rollout.py`: generation and reward computation.
3. `slime/rollout/rm_hub/__init__.py`: custom-RM loading and dispatch.
4. `slime/backends/megatron_utils/update_weight/`: actual policy transport/publication.

Explain which process owns optimizer state, how rollout log probabilities are associated with the behavior policy, and when SGLang may accept the next batch. Record actual update/sync logs before calling this milestone complete. Local `load_state_dict` tests cannot establish correctness of distributed weight conversion or transport.

## Completion criteria for both stacks

Run a short baseline, our task, a checkpoint/resume drill, a rollout-worker failure drill and an extended bounded soak. Record config and logs using [experiment-template.md](experiment-template.md). Confirm held-out evaluation is not sampled from training data. Inspect several raw answers, including invalid formats, and explain why high task reward is not broad model capability.

The completed local labs demonstrate practical Python/PyTorch and objective implementation. Only the GPU artifacts above can establish hands-on operation of the named distributed training stacks.
