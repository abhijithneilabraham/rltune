"""No-network failure drill: bounded async workers and retry accounting."""

import asyncio
import json

from .operations import collect


async def main():
    attempts = {}

    async def generate(job):
        attempts[job] = attempts.get(job, 0) + 1
        if job == "transient" and attempts[job] == 1:
            raise ConnectionError("simulated rollout service disconnect")
        if job == "timeout":
            await asyncio.sleep(1)
        return {"completion": "3", "policy_version": 0}

    results = await collect(
        ["normal", "transient", "timeout"], generate, concurrency=2, timeout=0.01, retries=1
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
