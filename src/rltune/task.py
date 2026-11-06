"""A strict outcome verifier shared by the lab and external reward hooks."""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Verdict:
    reward: float
    reason: str


def verify(response: str, answer: int) -> Verdict:
    # Full-match prevents rewarding an answer embedded in an otherwise invalid response.
    if not isinstance(response, str) or len(response) > 32:
        return Verdict(0.0, "invalid_type_or_length")
    if re.fullmatch(r"(?:0|[1-9][0-9]?)", response) is None:
        return Verdict(0.0, "invalid_format")
    return Verdict(
        float(int(response) == answer), "correct" if int(response) == answer else "wrong"
    )


def examples(split: str):
    if split not in {"train", "eval"}:
        raise ValueError("split must be train or eval")
    # Hold out operand PAIRS, rather than randomly splitting repeated rollouts.
    return [
        (a, b) for a in range(10) for b in range(10) if ((a * 10 + b) % 5 == 0) == (split == "eval")
    ]


def prompt(a: int, b: int) -> str:
    return f"What is {a} + {b}? Return only the integer, with no spaces or explanation."
