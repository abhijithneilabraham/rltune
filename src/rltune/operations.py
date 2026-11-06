"""Bounded async collection, explicit failure accounting, atomic snapshots."""

import asyncio
import os
import tempfile
from pathlib import Path

import torch


async def collect(jobs, generate, *, concurrency=4, timeout=5.0, retries=1):
    """Retry only timeout/connection failures. Preserve input order and bound active tasks."""
    if concurrency < 1 or timeout <= 0 or retries < 0:
        raise ValueError("invalid collection limits")
    iterator = iter(enumerate(jobs))
    results = {}

    async def worker():
        for index, job in iterator:
            for attempt in range(retries + 1):
                try:
                    async with asyncio.timeout(timeout):
                        value = await generate(job)
                    results[index] = {"value": value, "attempts": attempt + 1, "error": None}
                    break
                except (TimeoutError, ConnectionError) as exc:
                    if attempt == retries:
                        results[index] = {
                            "value": None,
                            "attempts": attempt + 1,
                            "error": type(exc).__name__,
                        }

    async with asyncio.TaskGroup() as group:
        for _ in range(concurrency):
            group.create_task(worker())
    return [results[i] for i in range(len(results))]


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".checkpoint-")
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


def synchronize(learner, rollout, version):
    """Local state-dict transfer. A distributed transport must provide the same contract."""
    rollout.load_state_dict(learner.state_dict())
    rollout.eval()
    return version


def require_version(rollout_version, learner_version, max_lag=0):
    lag = learner_version - rollout_version
    if lag < 0 or lag > max_lag:
        raise ValueError(f"policy lag {lag} outside [0, {max_lag}]")
