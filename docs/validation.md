# Implementation validation

Validated locally on 6 November 2026, macOS arm64, Python 3.12.3, PyTorch 2.15.0, CPU.

| Check | Result | Scope |
|---|---|---|
| Core PyTorch/operations/HTTP-client tests | 23 passed | All four updates, exact restart, gradient direction, verifier attacks, failed serialization, competing writers, stale policy, bounded timeout/retry accounting, HTTP error accounting |
| Optional real verifiers integration test | 1 passed | Real Taskset, TaskData and Trace construction; split separation; reward behavior, including raw whitespace |
| verifiers CLI setup | 2/2 valid | Subprocess runtime, no hosted service or model call |
| verifiers CLI noop | 2/2 valid | Empty completion does not receive a passing reward |
| Four 200-step learning runs | Completed | Single seed; metrics in local-results.json |
| Ruff lint and format | Passed | Python source, tests and integration adapters |
| prime-rl recipe TOML | Parsed | Syntax only; framework schema/GPU execution pending |
| Documentation links and code fences | Passed | Local paths and balanced code fences |

Commands:

```bash
uv sync --extra dev --locked
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run python scripts/compare_methods.py --output runs/comparison

uv sync --extra dev --extra prime --locked
uv pip install --no-deps -e integrations/prime
uv run --no-sync pytest -q
uv run --no-sync vf-validate rltune-arithmetic -n 2 --only-setup \
  --runtime.type subprocess -o runs/verifiers-validation
uv run --no-sync vf-validate rltune-arithmetic -n 2 --only-noop \
  --runtime.type subprocess -o runs/verifiers-noop
```

Without the optional integration installed, the core run reports 23 passed and one skipped module. With it installed, the full run reports 24 passed. CI includes separate core and verifiers jobs; the hosted CI result must be checked after pushing.

Not executed: slime or prime-rl GPU training, Megatron checkpoint conversion, distributed weight sync, real SGLang HTTP evaluation, Hub publication, multi-node restart, or long-running soak. The HTTP evaluator is tested with an `httpx.MockTransport` representing the external service. The local async failure drill is a controlled operational exercise. These limits are intentional and kept separate from completed evidence.
