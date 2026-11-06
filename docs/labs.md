# Learning sequence

Keep an experiment record for every lab. Make one change at a time, predict its effect, then compare against the unmodified run.

## 1. Task and verifier

Read `task.py`. Explain why `3`, `03`, `3\n`, and `the answer is 3` do not receive the same score. Run the verifier tests. Add an adversarial response that includes the correct answer but should receive zero. Do not put labels into the model input.

Acceptance: explain the verifier's contract, distinguish correctness from format, and show train/eval operand pairs do not overlap. Exact answer grading does not verify the reasoning process, language ability, or robustness outside the finite task.

## 2. SFT

Run the README SFT command. Print logits, targets, gradient norms and the first/final losses. Repeat with a smaller learning rate by calling `train(Config(method='sft', lr=0.001), 200, Path('runs/sft-small-lr'))` from Python. Explain why training more steps and changing the learning rate are separate changes.

Acceptance: an actual checkpoint, metric history, and an explanation of cross-entropy and Adam state. Repeat two seeds and report the spread. No requirement to manufacture a particular accuracy.

## 3. DPO

Read the chosen/rejected construction and `dpo_loss`. Run DPO. Inspect the policy/reference preference margin. Deliberately swap chosen/rejected in a separate experimental branch and predict the gradient direction before testing it.

Acceptance: explain synthetic preference construction, frozen reference, beta, and why a lower preference loss can coexist with worse exact-match accuracy.

## 4. GRPO

Run GRPO and inspect `[batch, group]` rewards. Count groups with no correct answers and groups with all identical rewards. Change `group_size` through the Python Config API. Group size changes sampling cost as well as reward diversity.

Acceptance: explain normalized advantages, clipped importance ratios, KL, and why more samples are not always more useful. Record train reward and held-out accuracy separately.

## 5. PPO

Run PPO. Inspect rollout values and terminal rewards, then the critic loss. Compare with GRPO at the same number of sampled answers, not merely the same number of steps.

Acceptance: explain the value baseline and one-step return. Sketch how token-sequence PPO would add terminal masks and GAE. The current code is not a multi-step agent trainer.

## 6. Harness operations

```bash
uv run python -m rltune.rollout_demo
uv run pytest -q tests/test_lab.py -k 'resume or checkpoint or staleness or collection'
```

The collector bounds concurrent jobs, times out stalled work and retries only connection/timeout errors. It returns an explicit failure record rather than silently giving a failed rollout reward zero. A malformed request raises instead of retrying forever. Real backends need backoff, request IDs, cancellation propagation and deduplication before retries can be called safe.

Train 20 steps; resume to 40. Compare with a fresh 40-step run. Delete only `metrics.json`, then resume: the checkpoint's history reconstructs the metrics. Keep the checkpoint intact. Test a stale rollout version. Explain why equal version numbers without a completed weight transfer are insufficient in a distributed implementation.

Acceptance: show failure records, exact restart equivalence, and one successful version acknowledgment. Read the operations runbook before the GPU milestone.

## 7. Operate the external stacks

Follow [upstream-stacks.md](upstream-stacks.md). First validate the verifiers taskset locally. Then use a supported Linux NVIDIA host for each framework. Start from the pinned framework's own working recipe, establish its baseline, and only then replace its task/reward.

Acceptance requires actual artifacts: revisions, image digest, launch config, train/eval logs, rollout policy versions, weight-sync timings, checkpoint restore logs and an extended stability run. Mark each requirement pending until you have that evidence. Running this local repository does not complete this milestone.
