import asyncio
from pathlib import Path

import pytest
import torch

from rltune.losses import clipped_policy_loss, dpo_loss, group_advantages
from rltune.operations import collect, require_version
from rltune.task import examples, verify
from rltune.training import Config, train


@pytest.mark.parametrize(
    "response", ["3 blah", "<answer>3</answer>", "03", "3\n", "+3", "３", "", "3" * 100]
)
def test_verifier_rejects_reward_hacking(response):
    assert verify(response, 3).reward == 0


def test_verifier_and_split():
    assert verify("3", 3).reward == 1
    assert verify("4", 3).reward == 0
    assert not set(examples("train")) & set(examples("eval"))
    assert len(examples("train")) + len(examples("eval")) == 100


def test_equal_reward_groups_have_zero_finite_advantages():
    a = group_advantages(torch.ones(2, 4))
    assert torch.equal(a, torch.zeros_like(a))


def test_clipping_and_gradient_direction():
    logp = torch.tensor([0.0], requires_grad=True)
    loss = clipped_policy_loss(logp, torch.tensor([0.0]), torch.tensor([1.0]))
    loss.backward()
    assert logp.grad.item() < 0  # descent increases probability of rewarded action
    clipped = clipped_policy_loss(torch.tensor([2.0]), torch.zeros(1), torch.ones(1))
    assert clipped.item() == pytest.approx(-1.2)


def test_dpo_prefers_chosen_action():
    chosen = torch.tensor([0.0], requires_grad=True)
    dpo_loss(chosen, torch.zeros(1), torch.zeros(1), torch.zeros(1)).backward()
    assert chosen.grad.item() < 0


@pytest.mark.parametrize("method", ["sft", "dpo", "grpo", "ppo"])
def test_resume_matches_uninterrupted_training(tmp_path: Path, method):
    c = Config(method=method, batch_size=8, group_size=4)
    full = train(c, 6, tmp_path / "full")
    train(c, 3, tmp_path / "resume")
    resumed = train(c, 6, tmp_path / "resume", resume=True)
    assert full == resumed
    a = torch.load(tmp_path / "full/checkpoint.pt", weights_only=True)
    b = torch.load(tmp_path / "resume/checkpoint.pt", weights_only=True)
    for key in a["model"]:
        assert torch.equal(a["model"][key], b["model"][key])
    with pytest.raises(ValueError, match="config differs"):
        train(Config(method=method, seed=99), 8, tmp_path / "resume", resume=True)


def test_policy_staleness_rejected():
    require_version(3, 3)
    with pytest.raises(ValueError, match="lag"):
        require_version(2, 3)
    with pytest.raises(ValueError, match="lag"):
        require_version(4, 3)


def test_bounded_collection_retries_and_accounts_for_failures():
    active, peak = 0, 0
    attempts = {}

    async def generate(job):
        nonlocal active, peak
        attempts[job] = attempts.get(job, 0) + 1
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0)
            if job == 1 and attempts[job] == 1:
                raise ConnectionError("transient")
            if job == 2:
                await asyncio.sleep(1)
            return job * 2
        finally:
            active -= 1

    results = asyncio.run(collect(range(4), generate, concurrency=2, timeout=0.01, retries=1))
    assert peak <= 2 and active == 0
    assert results[1]["value"] == 2 and results[1]["attempts"] == 2
    assert results[2]["error"] == "TimeoutError" and results[2]["attempts"] == 2
    assert results[3]["value"] == 6


def test_programming_errors_are_not_silently_retried():
    async def broken(job):
        raise ValueError("invalid request")

    with pytest.raises(ExceptionGroup):
        asyncio.run(collect([1], broken))


def test_sft_learns_training_task(tmp_path):
    history = train(Config(method="sft"), 100, tmp_path / "learn")
    assert history[-1]["loss"] < history[0]["loss"] * 0.5
    assert history[-1]["eval_accuracy"] > history[0]["initial_eval"]


def test_failed_checkpoint_write_preserves_previous_snapshot(tmp_path):
    from rltune.operations import atomic_save

    checkpoint = tmp_path / "checkpoint.pt"
    atomic_save({"step": 3}, checkpoint)
    with pytest.raises(AttributeError):
        atomic_save({"unserializable": lambda x: x}, checkpoint)
    assert torch.load(checkpoint, weights_only=True) == {"step": 3}
    assert not list(tmp_path.glob(".checkpoint-*"))


def test_run_directory_cannot_have_two_writers(tmp_path):
    import fcntl

    run = tmp_path / "shared"
    with (tmp_path / ".shared.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="another trainer"):
            train(Config(), 1, run)
