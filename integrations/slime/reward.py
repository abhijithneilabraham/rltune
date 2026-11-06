"""slime --custom-rm-path integrations.slime.reward.reward.

Install rltune and put this repository on PYTHONPATH on ALL reward workers.
Supports per-sample and batched custom-RM dispatch at the recorded slime revision.
"""

from rltune.task import verify


async def reward(args, sample, **kwargs):
    if isinstance(sample, list):
        return [verify(item.response, int(item.label)).reward for item in sample]
    return verify(sample.response, int(sample.label)).reward
