"""Optional: real verifiers objects, no inference service or credentials."""

import asyncio

import pytest

vf = pytest.importorskip("verifiers.v1")
pytest.importorskip("rltune_arithmetic")
from rltune_arithmetic.taskset import ArithmeticConfig, ArithmeticTaskset


def test_real_taskset_splits_and_reward_trace():
    train = ArithmeticTaskset(ArithmeticConfig(split="train")).load()
    evaluate = ArithmeticTaskset(ArithmeticConfig(split="eval")).load()
    assert len(train) == 80 and len(evaluate) == 20
    assert not {t.data.prompt for t in train} & {t.data.prompt for t in evaluate}
    task = train[0]
    for response, expected in [
        (str(task.data.answer), 1.0),
        ("wrong", 0.0),
        (str(task.data.answer) + "\n", 0.0),
    ]:
        trace = vf.Trace(
            task={"type": "rltune-arithmetic", "data": task.data},
            agent={"config": {}},
            root_reply=response,
        )
        assert asyncio.run(task.exact_integer(trace)) == expected
