import argparse
import json
from pathlib import Path

from .training import Config, train


def main():
    parser = argparse.ArgumentParser(description="CPU PyTorch post-training lab")
    parser.add_argument("method", choices=["sft", "dpo", "grpo", "ppo"])
    parser.add_argument(
        "--steps", type=int, default=200, help="total steps, including resumed steps"
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    history = train(
        Config(method=args.method, seed=args.seed), args.steps, args.output, args.resume
    )
    print(json.dumps(history[-1], indent=2))


if __name__ == "__main__":
    main()
