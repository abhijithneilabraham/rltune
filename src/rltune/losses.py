"""One-action objectives. Each sampled answer is a complete one-step episode."""

import torch
import torch.nn.functional as F


def group_advantages(rewards: torch.Tensor) -> torch.Tensor:
    if rewards.ndim != 2 or rewards.shape[1] < 2:
        raise ValueError("GRPO needs [prompts, group_size >= 2]")
    return (rewards - rewards.mean(-1, keepdim=True)) / rewards.std(
        -1, keepdim=True, unbiased=False
    ).clamp_min(1e-6)


def clipped_policy_loss(logp, old_logp, advantage, clip=0.2):
    ratio = (logp - old_logp.detach()).exp()
    return -torch.minimum(
        ratio * advantage.detach(), ratio.clamp(1 - clip, 1 + clip) * advantage.detach()
    ).mean()


def dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta=0.1):
    margin = (chosen - rejected) - (ref_chosen - ref_rejected).detach()
    return -F.logsigmoid(beta * margin).mean()


def categorical_kl(logits, reference_logits):
    logp = logits.log_softmax(-1)
    logq = reference_logits.detach().log_softmax(-1)
    return (logp.exp() * (logp - logq)).sum(-1).mean()
