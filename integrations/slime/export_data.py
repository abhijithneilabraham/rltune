"""Write JSONL input for slime; labels remain outside the model prompt."""

import json
from pathlib import Path

from rltune.task import examples, prompt


def main():
    target = Path("runs/slime-data")
    target.mkdir(parents=True, exist_ok=True)
    for split in ("train", "eval"):
        rows = [{"prompt": prompt(a, b), "label": str(a + b)} for a, b in examples(split)]
        (target / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


if __name__ == "__main__":
    main()
