"""Real CPU PyTorch updates; intentionally no transformer or distributed backend."""

import copy
import fcntl
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.distributions import Categorical

from .losses import categorical_kl, clipped_policy_loss, dpo_loss, group_advantages
from .operations import atomic_save, require_version, synchronize
from .task import examples, verify


@dataclass(frozen=True)
class Config:
    method: str = "grpo"
    seed: int = 7
    batch_size: int = 32
    group_size: int = 16
    lr: float = 0.01
    kl_coef: float = 0.01
    epochs: int = 2

    def validate(self):
        if self.method not in {"sft", "dpo", "grpo", "ppo"}:
            raise ValueError("unknown method")
        if self.batch_size < 1 or self.group_size < 2 or self.epochs < 1:
            raise ValueError("invalid batch/group/epochs")
        if not 0 < self.lr < 1 or not 0 <= self.kl_coef < 10:
            raise ValueError("invalid learning rate or KL coefficient")


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.actor = nn.Linear(64, 19)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        h = self.body(x / 9.0)
        return self.actor(h), self.critic(h).squeeze(-1)


def dataset(split):
    pairs = examples(split)
    return torch.tensor(pairs, dtype=torch.float32), torch.tensor([a + b for a, b in pairs])


@torch.no_grad()
def evaluate(model):
    x, y = dataset("eval")
    return (model(x)[0].argmax(-1) == y).float().mean().item()


def _train(config: Config, steps: int, output: Path, resume=False):
    config.validate()
    if steps < 1:
        raise ValueError("steps must be positive")
    output = Path(output)
    checkpoint = output / "checkpoint.pt"
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError("output is not empty; use --resume or a new directory")
    # One thread avoids tiny-matrix thread overhead and makes CPU continuation reproducible.
    torch.set_num_threads(1)
    torch.manual_seed(config.seed)
    model = Policy()
    reference = copy.deepcopy(model).requires_grad_(False)
    rollout = copy.deepcopy(model).requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    version, history = 0, []
    if resume:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state["schema"] != 1:
            raise ValueError("unsupported checkpoint schema")
        if state["config"] != asdict(config):
            raise ValueError("resume config differs from checkpoint")
        model.load_state_dict(state["model"])
        reference.load_state_dict(state["reference"])
        optimizer.load_state_dict(state["optimizer"])
        version, history = state["version"], state["history"]
        torch.set_rng_state(state["rng"])
    initial_eval = history[0]["initial_eval"] if history else evaluate(model)
    x_all, y_all = dataset("train")
    for step in range(version, steps):
        idx = torch.randint(len(x_all), (config.batch_size,))
        x, y = x_all[idx], y_all[idx]
        reward_mean, clip_fraction = None, 0.0
        if config.method in {"sft", "dpo"}:
            logits, _ = model(x)
            if config.method == "sft":
                loss = nn.functional.cross_entropy(logits, y)
            else:
                rejected = (y + torch.randint(1, 19, y.shape)) % 19
                lp = logits.log_softmax(-1)
                with torch.no_grad():
                    rp = reference(x)[0].log_softmax(-1)
                rows = torch.arange(len(y))
                loss = dpo_loss(lp[rows, y], lp[rows, rejected], rp[rows, y], rp[rows, rejected])
            epochs = 1
        else:
            rollout_version = synchronize(model, rollout, step)
            require_version(rollout_version, step)
            with torch.no_grad():
                old_logits, old_value = rollout(x)
                actions = Categorical(logits=old_logits).sample((config.group_size,)).T
                old_lp = old_logits.log_softmax(-1).gather(1, actions)
                rewards = torch.tensor(
                    [
                        [verify(str(a), int(target)).reward for a in row]
                        for row, target in zip(actions.tolist(), y.tolist())
                    ]
                )
                advantages = (
                    group_advantages(rewards)
                    if config.method == "grpo"
                    else rewards - old_value[:, None]
                )
                reward_mean = rewards.mean().item()
            epochs = config.epochs
        for _ in range(epochs):
            if config.method in {"grpo", "ppo"}:
                logits, value = model(x)
                lp = logits.log_softmax(-1).gather(1, actions)
                loss = clipped_policy_loss(lp, old_lp, advantages)
                loss = loss + config.kl_coef * categorical_kl(logits, reference(x)[0])
                if config.method == "ppo":
                    loss = loss + 0.5 * (value[:, None] - rewards).square().mean()
                clip_fraction = (((lp - old_lp).exp() - 1).abs() > 0.2).float().mean().item()
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite loss; last complete checkpoint preserved")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError("non-finite parameters; checkpoint preserved")
        with torch.no_grad():
            kl = categorical_kl(model(x)[0], reference(x)[0]).item()
        row = {
            "step": step + 1,
            "policy_version": step + 1,
            "loss": loss.item(),
            "reward": reward_mean,
            "eval_accuracy": evaluate(model),
            "reference_kl": kl,
            "clip_fraction": clip_fraction,
            "grad_norm": float(grad_norm),
            "initial_eval": initial_eval,
        }
        history.append(row)
        # History is in the checkpoint: metrics can be reconstructed after a crash.
        atomic_save(
            {
                "schema": 1,
                "config": asdict(config),
                "model": model.state_dict(),
                "reference": reference.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng": torch.get_rng_state(),
                "version": step + 1,
                "history": history,
            },
            checkpoint,
        )
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "metrics.json.tmp"
    temporary.write_text(json.dumps(history, indent=2) + "\n")
    temporary.replace(output / "metrics.json")
    return history


def train(config: Config, steps: int, output: Path, resume=False):
    """Allow one writer per run directory on Linux/macOS."""
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / f".{output.name}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another trainer owns this run directory") from exc
        return _train(config, steps, output, resume)
