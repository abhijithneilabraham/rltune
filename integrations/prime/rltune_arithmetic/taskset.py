"""Adapter for verifiers v1, based on the pinned upstream Taskset contract."""

from typing import Literal

import verifiers.v1 as vf

from rltune.task import examples, prompt, verify


class ArithmeticData(vf.TaskData):
    answer: int


class ArithmeticTask(vf.Task[ArithmeticData]):
    @vf.stop
    async def single_turn(self, trace: vf.Trace) -> bool:
        return trace.num_turns >= 1

    @vf.reward(weight=1.0)
    async def exact_integer(self, trace: vf.Trace) -> float:
        response = trace.root_reply
        if response is None:
            messages = trace.assistant_messages
            response = messages[-1].content if messages else ""
        return verify(response or "", self.data.answer).reward


class ArithmeticConfig(vf.TasksetConfig):
    split: Literal["train", "eval"] = "train"


class ArithmeticTaskset(vf.Taskset[ArithmeticTask, ArithmeticConfig]):
    def load(self) -> list[ArithmeticTask]:
        return [
            ArithmeticTask(ArithmeticData(prompt=prompt(a, b), answer=a + b), self.config.task)
            for a, b in examples(self.config.split)
        ]
