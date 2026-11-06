"""Evaluate held-out prompts on an explicitly supplied OpenAI-compatible server."""

import argparse
import asyncio
import json
import os
from pathlib import Path

import httpx

from .operations import collect
from .task import examples, prompt, verify


async def evaluate_server(client, model, concurrency=4):
    async def generate(pair):
        a, b = pair
        try:
            response = await client.post(
                "chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt(a, b)}],
                    "temperature": 0,
                    "max_tokens": 32,
                },
            )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise ConnectionError(type(exc).__name__) from exc
        if response.status_code != 200:
            return {"ok": False, "error": f"HTTP {response.status_code}"}
        try:
            payload = response.json()
            text = payload["choices"][0]["message"]["content"]
            if not isinstance(text, str):
                raise TypeError("non-text response")
        except (KeyError, IndexError, TypeError, ValueError):
            return {"ok": False, "error": "invalid_completion_schema"}
        verdict = verify(text, a + b)
        return {
            "ok": True,
            "prompt": prompt(a, b),
            "completion": text,
            "reward": verdict.reward,
            "reason": verdict.reason,
        }

    records = await collect(
        examples("eval"), generate, concurrency=concurrency, timeout=35.0, retries=0
    )
    successful = [r["value"] for r in records if r["value"] and r["value"]["ok"]]
    return {
        "model": model,
        "attempted": len(records),
        "completed": len(successful),
        "failed": len(records) - len(successful),
        "accuracy_on_completed": (
            sum(r["reward"] for r in successful) / len(successful) if successful else None
        ),
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="Include /v1/; must be a trusted server")
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new file")
    key = os.environ.get(args.api_key_env)
    headers = {"Authorization": f"Bearer {key}"} if key else {}

    async def run():
        async with httpx.AsyncClient(
            base_url=args.base_url.rstrip("/") + "/", headers=headers, timeout=30.0
        ) as client:
            return await evaluate_server(client, args.model, args.concurrency)

    result = asyncio.run(run())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
