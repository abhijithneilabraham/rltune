"""Reproduce the small learning check; results are not a benchmark ranking."""

import argparse
import json
import platform
from dataclasses import asdict
from pathlib import Path

import torch

from rltune.training import Config, train

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--output", type=Path, default=Path("runs/comparison"))
args = parser.parse_args()
results = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "platform": platform.system() + "-" + platform.machine(),
    "device": "cpu",
    "steps": args.steps,
    "seed": args.seed,
    "eval_pairs": 20,
    "runs": {},
}
for method in ("sft", "dpo", "grpo", "ppo"):
    config = Config(method=method, seed=args.seed)
    history = train(config, args.steps, args.output / method)
    results["runs"][method] = {"config": asdict(config), "first": history[0], "last": history[-1]}
(args.output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results, indent=2))
