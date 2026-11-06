# How the updates work

Read `src/rltune/losses.py`, then put a breakpoint in `training.py` before `loss.backward()`.

For a batch of B prompts, logits have shape `[B, 19]`: one score per possible sum 0–18. `log_softmax` gives log probabilities. Group sampling produces actions, rewards, and behavior log probabilities with shape `[B, G]`. The prompt contains two operands, never the expected answer. The verifier sees the answer separately.

## SFT

`loss = -mean(log p(correct_answer | operands))`

We have a labeled correct answer for every training pair. Cross-entropy increases its probability. `backward()` accumulates gradients; Adam uses gradients and running moments; `zero_grad(set_to_none=True)` prevents accidental accumulation. There is no rollout reward or reference model in the SFT objective.

Exercise: inspect `actor.weight.grad` and change the batch size. Explain why a lower training loss need not imply a higher held-out accuracy.

## DPO

For chosen action y+ and rejected action y−:

```text
margin = [log p(y+) - log p(y-)] - [log pref(y+) - log pref(y-)]
loss = -log sigmoid(beta * margin)
```

The reference is frozen at initialization. In this lab, chosen is the true sum and rejected is a uniformly sampled wrong sum; these are synthetic preferences, not human preference data. DPO uses offline pairs; no RL rollout or value function is needed. The test checks gradient direction. Beta defaults to 0.1 in the objective.

A language-model implementation sums response-token log probabilities and masks prompt/padding tokens. This lab has one action, so that sum has one term. Length bias, truncation and response masking are absent here and must be tested when moving to LLMs.

## GRPO

For each prompt sample G answers from the rollout policy:

```text
A_i = (reward_i - group_mean) / max(group_std, 1e-6)
ratio_i = exp(current_logp_i - behavior_logp_i)
Lpolicy = -mean(min(ratio_i * A_i, clip(ratio_i, 0.8, 1.2) * A_i))
L = Lpolicy + kl_coef * KL(current || reference)
```

The behavior log probabilities and advantages are detached. Otherwise gradients would flow into the target that is supposed to remain fixed. Population standard deviation (`unbiased=False`) handles finite groups without a sample-correction surprise. If all rewards match, advantages are zero. The reference KL can still produce a gradient after the policy has moved away from reference.

The lab computes exact categorical KL across all 19 actions. LLM stacks commonly use sampled estimators and token-level weighting; do not assume those are numerically identical. Two optimization epochs reuse one rollout batch, making behavior/current policy divergence visible. There is no learned critic for GRPO.

## PPO

These are one-step contextual-bandit episodes, so return = terminal reward. The rollout critic supplies a detached baseline:

```text
A_i = reward_i - V_behavior(prompt)
L = clipped_policy_loss + 0.5 * mean((V_current - reward_i)^2)
    + kl_coef * KL(current || reference)
```

The actor and critic share a trunk and have separate output heads. PPO adds a critic loss; GRPO estimates a baseline from the group. Multi-step PPO needs bootstrapping, terminal masks and often generalized advantage estimation. This lab deliberately does not implement a token-sequence GAE algorithm under a one-step name.

## Three policies, three purposes

| Object | Updated when | Purpose |
|---|---|---|
| Learner | Each optimizer step | Policy being improved |
| Rollout/behavior | Before each sampled batch | Actions and fixed old log probabilities |
| Reference | Never, including after resume | KL anchor or DPO log-ratio anchor |

Copying the learner into the reference each step would change the objective. Using current log probabilities as the old policy after an update would hide policy drift. Recomputing rewards on a different prompt would corrupt credit assignment.

All methods clip gradient norm to 1.0 and refuse non-finite loss, gradient norm, or updated parameters. Clipping limits magnitude; it does not prove that learning is stable or reward design is good.
