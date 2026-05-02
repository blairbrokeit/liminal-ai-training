# Contributing

Thanks for your interest. This is a small open-source project — pull requests are welcome and read in good faith.

## Quick start for contributors

```bash
git clone https://github.com/blairbrokeit/liminal-ai-training.git
cd liminal-ai-training
pip install -e .[test]
```

The `[test]` extra brings in pytest. Run the (currently small) test suite with:

```bash
pytest
```

## What kinds of changes are welcome

- **Bug fixes** — always.
- **New environments** — see [`docs/BUILDING_ENVIRONMENTS.md`](docs/BUILDING_ENVIRONMENTS.md). New `LiminalEnvironment` subclasses with novel mechanics are great.
- **New NPC strategies** — `liminal/npc.py` has three strategies; more are interesting if they generate genuinely different preference pairs.
- **More benchmarks** — TruthfulQA is in. MMLU, GSM8K, HumanEval would all be good.
- **Better task generation** — the auto-task generator (`liminal/autotasks.py`) can always be sharper.
- **Documentation** — install issues, model-specific tips, new tutorials.
- **Adapter exports for new model formats** — GGUF, MLX, ONNX.

## What probably won't be merged

- Changes that swap the trainer to a non-DPO objective without a clear reason. DPO is the chosen contract.
- Changes that hard-couple the trainer to a single LLM provider. The judge and NPCs are pluggable on purpose.
- Pure refactors with no functional change and no measurable improvement to readability or testability.
- Removing the `null-agent` integration. The bridge is small and self-contained; leave it.

## Pull request checklist

Before opening a PR:

- [ ] `pytest` passes locally
- [ ] No new files committed by accident (check `git status`)
- [ ] No secrets, API keys, or `.env` content in any commit
- [ ] No model weights, adapters, or `models/` / `adapters/` directories committed
- [ ] No metrics output committed (`metrics/`, `*.jsonl` of session logs)
- [ ] Imports use `from liminal.X import Y`, not `from src.X import Y`
- [ ] If you added a new module, add a one-line entry to `liminal/__init__.py`'s docstring map
- [ ] If you changed CLI flags, update both the README and `docs/GETTING_STARTED.md`

## Commit messages

Plain English, imperative mood, ~72 char first line. Body describes the *why* if it isn't obvious. Examples in the existing log.

## Reporting issues

[Issues](https://github.com/blairbrokeit/liminal-ai-training/issues) are open. Please include:

- Python version (`python --version`)
- OS
- The exact command that failed
- The full error traceback
- For training-quality issues: which model, which task set, how many loops, the metrics file if you have it

## Companion repository

If your contribution touches the bridge with [`blairbrokeit/null-agent`](https://github.com/blairbrokeit/null-agent), please flag that in the PR description. The bridge is a one-way file contract — please do not introduce a runtime import dependency from this repo into `null-agent`.

## License

By contributing you agree your contributions will be released under this project's [MIT license](LICENSE).
