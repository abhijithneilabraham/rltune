"""Fetch pinned source checkouts only. Does not install or launch GPU jobs."""

import json
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
lock = json.loads((root / "upstream.lock.json").read_text())
for name in ("slime", "prime-rl", "verifiers"):
    target = root / ".external" / name
    spec = lock[name]
    if target.exists():
        head = subprocess.check_output(
            ["git", "-C", str(target), "rev-parse", "HEAD"], text=True
        ).strip()
        if head != spec["commit"]:
            raise SystemExit(f"{target} differs from pin; inspect it before replacing it")
    else:
        target.mkdir(parents=True)
        subprocess.run(["git", "init", str(target)], check=True)
        subprocess.run(
            ["git", "-C", str(target), "remote", "add", "origin", spec["url"]], check=True
        )
        subprocess.run(
            ["git", "-C", str(target), "fetch", "--depth=1", "origin", spec["commit"]], check=True
        )
        subprocess.run(["git", "-C", str(target), "checkout", "--detach", "FETCH_HEAD"], check=True)
    print(name, spec["commit"])
