# Operating and recovering a run

## Before launch

Record the source commit, dependency lock, seed, model/tokenizer revision, training data hash, evaluation split, optimizer settings and hardware. Start with one short run. Measure the number of generated samples, accepted samples, updates, and checkpoint writes. Keep training/evaluation prompts separate.

## Weight ownership and publication

The learner owns trainable weights. The rollout policy receives an immutable snapshot. Publish a new version only after every required tensor has arrived and workers acknowledge readiness. Do not mix shards from two versions. In this local lab `load_state_dict` is synchronous and an integer version is checked before training. There are no network workers or shard acknowledgments.

For an asynchronous GPU run, record the behavior version on every sample. Define a maximum allowed lag and reject or explicitly correct stale data. A timeout must not silently mark a partial transfer complete. Inspect framework-specific distributed transport instead of assuming this local copy implements it.

## Symptoms and response

| Signal | Investigate | Immediate action |
|---|---|---|
| Reward rises, eval does not | Leakage, format exploitation, easy prompts, overfit | Review raw responses and verifier; evaluate fixed held-out prompts |
| KL jumps or entropy collapses | LR, stale logprobs, excessive reuse, reward scale | Preserve checkpoint/logs; stop the run; compare old/current/reference distributions |
| Many identical-reward GRPO groups | Sparse rewards or deterministic behavior | Inspect per-group rewards; change difficulty or sampling in a separate experiment |
| Rollout timeout/connection errors | Queue depth, server health, request lengths | Bound concurrency; retry only transient errors with a finite budget |
| OOM | Token budget, longest sequence, KV cache, optimizer memory | Measure memory; change one allocation driver; resume last complete checkpoint |
| Hanging weight update | Rank readiness, collective mismatch, worker failure | Treat publish as incomplete; recover the worker group consistently |
| Non-finite loss/gradients | Precision, overflow, reward scale, corrupt data | Fail before checkpoint publication; inspect last good state |
| Different resumed trajectory | Missing RNG, optimizer, sampler state, nondeterministic kernels | Compare checkpoint contents and first divergent step |

## Checkpoint guarantees in this repository

A checkpoint is written to a temporary file, flushed, fsynced and atomically renamed in the same directory; the directory is then fsynced. The previous complete snapshot survives a failed serialization. A POSIX file lock prevents competing local writers. This targets Linux/macOS local filesystems. It does not claim the same durability for object storage or all network filesystems.

The checkpoint contains step, config, schema version, model, reference, optimizer, torch CPU RNG and history. No Python/NumPy RNG is used by training. Add their state if you introduce them. CUDA RNG, rank-local RNG, data-loader cursor, distributed shard manifests, LR scheduler and AMP scaler would also be needed when those components exist.

A process killed during a step can lose that step's work. Resume redoes it from the prior complete checkpoint. `metrics.json` is derived from checkpoint history and written at the end; it is not a live monitoring stream. Per-step saving is intentionally simple and expensive; a real training job should measure save overhead and choose an interval against acceptable lost work.

## Extended GPU stability exercise

After a successful short baseline, choose and record a bounded soak duration appropriate to your GPU budget. Track rolling reward/eval, reference KL, entropy, grad norm, tokens/s, queue depth, error rate, policy lag, memory and checkpoint duration. Intentionally restart one rollout worker, then the trainer at a checkpoint boundary. Verify recovery and record any lost or duplicated samples. A successful ten-step smoke test is not evidence of long-running stability.
