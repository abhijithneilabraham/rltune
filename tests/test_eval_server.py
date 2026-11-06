import asyncio
import json

import httpx

from rltune.eval_server import evaluate_server


def test_server_eval_counts_errors_separately_from_wrong_answers():
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        body = json.loads(request.content)
        assert "answer" not in body
        assert body["model"] == "test-policy"
        if calls == 1:
            return httpx.Response(503)
        return httpx.Response(200, json={"choices": [{"message": {"content": "wrong"}}]})

    async def run():
        async with httpx.AsyncClient(
            base_url="http://test/v1/", transport=httpx.MockTransport(handler)
        ) as client:
            return await evaluate_server(client, "test-policy")

    report = asyncio.run(run())
    assert report["attempted"] == 20
    assert report["completed"] == 19 and report["failed"] == 1
    assert report["accuracy_on_completed"] == 0.0
