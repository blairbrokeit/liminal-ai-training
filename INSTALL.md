# INSTALL

A dead-simple install guide. Three minutes from clone to first command.

## Requirements

- **Python 3.10 or newer** — check with `python --version`
- **An OpenAI API key** for the judge and NPCs — get one at [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- **A GPU** (recommended) — 6 GB VRAM is enough for Phi-3 Mini, 16 GB for Llama 8B. CPU works for a small smoke test but training is slow.

You do **not** need:
- Docker
- A Hugging Face account (only required for gated models like Llama)
- Anything beyond `pip`

## Install (three commands)

```bash
git clone https://github.com/blairbrokeit/liminal-ai-training.git
cd liminal-ai-training
pip install -e .
```

That's it. The `-e` (editable) flag installs the package in place so you can edit the code and re-run without reinstalling.

After install, two console commands are on your PATH:

- `liminal-train` — the training loop
- `liminal-evaluate` — base-vs-adapter comparison

Verify:

```bash
liminal-train --help
```

## Optional extras

```bash
# 4-bit quantized loading (saves a lot of VRAM, CUDA only):
pip install -e .[quantize]

# Test dependencies:
pip install -e .[test]
```

## Set your API key

Create a `.env` file at the repo root:

```bash
cp .env.example .env
```

Edit `.env`:

```
OPENAI_API_KEY=sk-your-key-here
```

The trainer reads `.env` automatically via `python-dotenv`.

## First run (smoke test)

The repository ships a tiny example task set so you can verify everything before downloading a real model:

```bash
# Use a small model so this runs on a laptop GPU
python scripts/download_model.py --model microsoft/Phi-3-mini-4k-instruct

# 5 loops on the example tasks — should take a few minutes
liminal-train \
  --model ./models/Phi-3-mini-4k-instruct \
  --tasks tasks/example.jsonl \
  --adapter ./adapters/smoke-test \
  --loops 5
```

If you see loop output and an adapter saved to `./adapters/smoke-test`, you are done. Move on to [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md) for the full workflow (custom tasks, benchmarks, dashboard, regression testing).

## Install from PyPI (not yet)

We have not pushed to PyPI yet. For now, install from GitHub:

```bash
pip install git+https://github.com/blairbrokeit/liminal-ai-training.git
```

## Common install issues

### `pip` is older than expected

Upgrade pip first:

```bash
python -m pip install --upgrade pip
```

### `torch` install is slow

It is. PyTorch is several hundred MB. This is a one-time cost. If you already have a working PyTorch install (e.g. inside a conda env), skip the implicit install:

```bash
pip install -e . --no-deps
pip install transformers peft trl accelerate openai datasets pyyaml python-dotenv rich python-json-logger flask huggingface-hub safetensors
```

### CUDA version mismatch

Install the matching PyTorch wheel first, then this package:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

### `bitsandbytes` fails on macOS

`bitsandbytes` is CUDA-only. It is in the `[quantize]` extra, so the base install does not pull it in. On macOS, just skip the `[quantize]` extra.

### `liminal-train` is "not on PATH" after install

`pip install --user` puts scripts in a user directory that may not be on your PATH. Either add it to PATH or use:

```bash
python -m liminal.train --help
```

`python -m liminal.train` and `liminal-train` are equivalent.

## Uninstall

```bash
pip uninstall liminal-ai-training
```

The cloned source directory is harmless to delete.
