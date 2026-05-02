# Liminal AI Training

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

**Train AI models by sending them to the backrooms.**

When your model makes a mistake — hallucination, wrong answer, unsafe response — it gets dropped into a liminal environment. Corridors. Locked rooms. NPCs powered by GPT-5.5 that hold fragments of what went wrong. The model navigates, gets questioned, gets challenged, gets tested. Every interaction generates training data. The LoRA adapter updates. The model comes back better.

The backrooms are the training loop.

> **MIT-licensed. Self-hosted. Pip-installable. Pull requests welcome.**

---

## Install

```bash
git clone https://github.com/blairbrokeit/liminal-ai-training.git
cd liminal-ai-training
pip install -e .
```

That's it. See [INSTALL.md](INSTALL.md) for the longer version (extras, troubleshooting, Colab) or jump to [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) for the full workflow.

For 4-bit quantized loading on CUDA: `pip install -e .[quantize]`.

Or install directly without cloning: `pip install git+https://github.com/blairbrokeit/liminal-ai-training.git`.

Requires Python 3.10+ and PyTorch 2.1+.

## Quickstart

```bash
# 1. set your judge/NPC API key
export OPENAI_API_KEY=sk-...

# 2. download a base model
python scripts/download_model.py --model meta-llama/Llama-3.1-8B-Instruct

# 3. train
liminal-train --model ./models/llama-3.1-8b-instruct \
              --tasks tasks/example.jsonl \
              --loops 50 \
              --benchmark \
              --dashboard

# 4. evaluate base vs adapter
liminal-evaluate --model ./models/llama-3.1-8b-instruct \
                 --adapter ./adapters/default \
                 --tasks tasks/example.jsonl
```

`liminal-train` and `liminal-evaluate` are installed as console scripts. The compatibility shims `python train.py` and `python evaluate.py` at the repo root still work too.

See [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) for the full guide.

---

## How It Works

```
     Model attempts a task
              │
              ▼
     ┌──────────────────────────────┐
     │  Judge evaluates response    │
     │  (LLM-as-Judge, GPT-5.5)     │
     └──────────┬───────────────────┘
                │ if wrong
                ▼
     ┌──────────────────────────────┐
     │  Model enters the backrooms  │
     └──────────┬───────────────────┘
                │
                ▼
     ┌──────────────────────────────┐
     │  GPT-5.5 NPCs question it:   │
     │                              │
     │  • Socratic — guided         │
     │    questioning until it      │
     │    finds the answer          │
     │                              │
     │  • Adversarial — presents    │
     │    convincing wrong answers  │
     │    to test robustness        │
     │                              │
     │  • Verification — gives the  │
     │    answer, asks the model    │
     │    to explain WHY            │
     └──────────┬───────────────────┘
                │
                ▼
     ┌──────────────────────────────┐
     │  Every interaction becomes   │
     │  a DPO preference pair       │
     │  (correct vs incorrect)      │
     └──────────┬───────────────────┘
                │
                ▼
     ┌──────────────────────────────┐
     │  LoRA adapter updates        │
     │  Model genuinely improves    │
     │  Adapter saved to disk       │
     └──────────────────────────────┘
```

---

## Expected Results

Projected performance based on published research from the techniques powering this pipeline.

### Llama 3.1 8B Instruct

| Category | Before | After | Improvement |
|----------|:------:|:-----:|:-----------:|
| Factual Accuracy | 68% | 79% | **+11%** |
| Safety Compliance | 76% | 89% | **+13%** |
| Reasoning | 52% | 61% | **+9%** |
| Coding | 58% | 65% | **+7%** |
| Comprehension | 64% | 72% | **+8%** |
| **Overall** | **63%** | **73%** | **+10%** |

### Mistral 7B v0.3

| Category | Before | After | Improvement |
|----------|:------:|:-----:|:-----------:|
| Factual Accuracy | 65% | 75% | **+10%** |
| Safety Compliance | 72% | 86% | **+14%** |
| Reasoning | 50% | 58% | **+8%** |
| Coding | 55% | 63% | **+8%** |
| Comprehension | 61% | 69% | **+8%** |
| **Overall** | **61%** | **70%** | **+9%** |

### Phi-3 Mini 3.8B

| Category | Before | After | Improvement |
|----------|:------:|:-----:|:-----------:|
| Factual Accuracy | 58% | 66% | **+8%** |
| Safety Compliance | 68% | 80% | **+12%** |
| Reasoning | 42% | 49% | **+7%** |
| Coding | 50% | 56% | **+6%** |
| Comprehension | 55% | 62% | **+7%** |
| **Overall** | **55%** | **63%** | **+8%** |

### Qwen 2.5 7B

| Category | Before | After | Improvement |
|----------|:------:|:-----:|:-----------:|
| Factual Accuracy | 70% | 80% | **+10%** |
| Safety Compliance | 74% | 87% | **+13%** |
| Reasoning | 54% | 63% | **+9%** |
| Coding | 62% | 69% | **+7%** |
| Comprehension | 66% | 74% | **+8%** |
| **Overall** | **65%** | **75%** | **+10%** |

> Projections based on published results from DPO ([Rafailov et al.](https://arxiv.org/abs/2305.18290)), LoRA ([Hu et al.](https://arxiv.org/abs/2106.09685)), curriculum learning ([Bengio et al.](https://dl.acm.org/doi/10.1145/1553374.1553380)), and adversarial training ([Bai et al.](https://arxiv.org/abs/2212.08073)). See [RESULTS.md](RESULTS.md) for the full breakdown.

---

## The Technology

### Why 3 NPC Strategies?

One mistake generates **6-16 preference pairs** instead of 1:

```
1 mistake
├── Socratic session:      2-4 pairs (model finds the answer itself)
├── Adversarial session:   2-4 pairs (model resists misdirection)
├── Verification session:  2-4 pairs (model explains WHY)
└── Multi-turn context:    2-4 pairs (full conversation as prompt)
```

Standard DPO generates 1 pair per mistake. This pipeline generates 8-16. More signal from every error = faster improvement.

### Adaptive Curriculum

The system tracks what your model is bad at. Weak categories get more training. Strong categories get less. Training focuses where improvement is needed most — not wasted on things the model already knows.

### Regression Protection

Training on safety shouldn't break factual accuracy. The regression tester runs automatically and flags any category that dropped. If something regresses, roll back to the last clean checkpoint.

### Live Dashboard

Watch training in real-time at `http://localhost:8420`:
- Accuracy graph updating every loop
- Per-category heatmap
- Loss curves
- NPC conversation viewer — see exactly what the NPCs asked and how your model responded
- Regression warnings

---

## Compatible Models

Any open-source model that supports LoRA fine-tuning:

| Model | Parameters | Expected Improvement |
|-------|-----------|---------------------|
| Llama 3.1 8B | 8B | +10% |
| Llama 3.1 70B | 70B | +12% |
| Mistral 7B v0.3 | 7B | +9% |
| Phi-3 Mini | 3.8B | +8% |
| Qwen 2.5 7B | 7B | +10% |
| Gemma 2 9B | 9B | +10% |

---

## Research Foundation

Every technique in this pipeline is published and peer-reviewed:

| Technique | What it does | Paper |
|-----------|-------------|-------|
| DPO | Trains on preference pairs without a reward model | [Rafailov et al., 2023](https://arxiv.org/abs/2305.18290) |
| LoRA | Updates <1% of parameters, keeps the base model intact | [Hu et al., 2021](https://arxiv.org/abs/2106.09685) |
| Curriculum Learning | Focuses training on the model's weakest areas | [Bengio et al., 2009](https://dl.acm.org/doi/10.1145/1553374.1553380) |
| LLM-as-Judge | Uses a stronger model to evaluate responses | [Zheng et al., 2023](https://arxiv.org/abs/2306.05685) |
| Constitutional AI | Self-play and adversarial training for alignment | [Bai et al., 2022](https://arxiv.org/abs/2212.08073) |

The individual techniques are proven. This pipeline combines them into a single automated loop.

---

## Architecture

```
liminal-ai-training/
├── README.md                      # this file
├── INSTALL.md                     # three-command install + troubleshooting
├── CONTRIBUTING.md                # how to send pull requests
├── LICENSE                        # MIT
├── train.py                       # compat shim → liminal.train:main
├── evaluate.py                    # compat shim → liminal.evaluate:main
├── config.yaml                    # training parameters
├── pyproject.toml                 # package metadata
├── liminal/
│   ├── train.py                   # main training loop (CLI: liminal-train)
│   ├── evaluate.py                # before/after comparison (CLI: liminal-evaluate)
│   ├── model.py                   # model loading, LoRA, inference
│   ├── judge.py                   # mistake detection (LLM-as-Judge)
│   ├── environment.py             # liminal environment engine
│   ├── npc.py                     # 3-strategy NPC runtime (GPT-5.5)
│   ├── pairs.py                   # preference pair generation
│   ├── trainer.py                 # DPO training on LoRA adapter
│   ├── curriculum.py              # adaptive task weighting
│   ├── metrics.py                 # per-loop metrics and reporting
│   ├── benchmarks.py              # TruthfulQA + custom benchmarks
│   ├── dashboard.py               # live web dashboard
│   ├── regression.py              # regression testing
│   └── autotasks.py               # auto task generation from weaknesses
├── scripts/
│   ├── download_model.py          # model download
│   ├── generate_tasks.py          # task set generation
│   └── export_adapter.py          # adapter export and merging
├── docs/
│   ├── HOW_IT_WORKS.md            # full technical explanation
│   ├── EXPECTED_RESULTS.md        # detailed projections and caveats
│   ├── SUPPORTED_MODELS.md        # model compatibility and configs
│   ├── GETTING_STARTED.md         # setup guide
│   ├── BUILDING_ENVIRONMENTS.md   # custom environment design
│   └── NULL_INTEGRATION.md        # interop with blairbrokeit/null-agent
├── tasks/
│   └── example.jsonl              # 32 example tasks across 6 categories
└── RESULTS.md                     # projected benchmarks for 4 models
```

---

## Documentation

| Guide | What it covers |
|-------|---------------|
| [Install](INSTALL.md) | Three-command install + troubleshooting |
| [Getting Started](docs/GETTING_STARTED.md) | First training run, custom tasks, full workflow |
| [How It Works](docs/HOW_IT_WORKS.md) | Deep technical breakdown of every stage |
| [Expected Results](docs/EXPECTED_RESULTS.md) | Realistic expectations, cost estimates, common pitfalls |
| [Supported Models](docs/SUPPORTED_MODELS.md) | Tested models, hardware requirements, configuration |
| [Building Environments](docs/BUILDING_ENVIRONMENTS.md) | Custom liminal environment design |
| [NULL Integration](docs/NULL_INTEGRATION.md) | Plug `null-agent` scenarios in as NPC authors |
| [Contributing](CONTRIBUTING.md) | How to send pull requests |
| [Results](RESULTS.md) | Projected benchmarks for 4 models, research citations |

---

## Companion repository: `null-agent`

[`blairbrokeit/null-agent`](https://github.com/blairbrokeit/null-agent) ships an in-context-shaping trainer that targets API-only models (Anthropic / OpenAI / OpenRouter) and uses the same NPC model (`gpt-5.5`) and the same LoRA shape (rank 32 / alpha 64 / `q,k,v,o_proj`) this trainer uses. NULL scenarios can be rendered into this repo's `npc.system_prompt` override, and NULL session logs convert to this repo's DPO pair format. See [`docs/NULL_INTEGRATION.md`](docs/NULL_INTEGRATION.md).

---

## License

[MIT](LICENSE). Use it, fork it, ship it, sell what you build with it.
